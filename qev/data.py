"""Reproducible supervised decision data, without reading Kev's locked test.

Public requests are converted by jaredpalmer/kev (Apache-2.0 code); the underlying
datasets retain their own licenses. New bilingual rules use executable labels.
Only upstream train/calibration/development partitions can be opened here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

from .api import SystemOneRequest, render, to_record

SPLITS = ("train", "calibration", "development")
PUBLIC_SOURCES = ("banking77", "boolq", "agnews", "mnli", "sst5", "yelp", "trec", "dbpedia14", "amazon", "imdb")
SUITE_REPOSITORY = "jaredpalmer/kev-suites"
SUITE_REVISION = "a3318ddc1f630c5673232efacd8123a84de3f480"
SUITE_PATH = "public-pool-v4"
UPSTREAM_CODE = "https://github.com/jaredpalmer/kev"


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def materialize(request: dict) -> dict:
    """Use exactly the serving renderer, then attach integer supervision labels."""
    clean = {"state": request["state"], "questions": {
        qid: {key: value for key, value in question.items() if key not in ("label", "src")}
        for qid, question in request["questions"].items()}}
    record, metadata = to_record(SystemOneRequest.model_validate(clean))
    for question, meta, (qid, original) in zip(record["questions"], metadata, request["questions"].items()):
        label = original["label"]
        if question["qtype"] == "noul":
            if type(label) is not bool:
                raise ValueError("noul training labels must be booleans")
            question["label"] = int(label)
        elif question["qtype"] == "choice":
            if label not in meta["keys"]:
                raise ValueError("choice training label is absent from criteria")
            question["label"] = meta["keys"].index(label)
        else:
            if type(label) is not int or not 0 <= label < len(question["options"]):
                raise ValueError("score label must be an in-range integer level")
            question["label"] = label
        question["src"] = original.get("src", request.get("_meta", {}).get("source", "unknown"))
    return record


def augment(request: dict, rng: random.Random, p_none: float = 0.10,
            p_none_distract: float = 0.12, p_distract: float = 0.15) -> dict:
    """Training-only option permutations and transparent none/distractor labels."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("invalid augmentation probabilities")
    result = copy.deepcopy(request)
    zh = request.get("_meta", {}).get("language") == "zh"
    none_text = "其他选项都不符合当前情况。" if zh else "None of the other options matches the state."
    distractor_text = "当前情况未提及的无关主题。" if zh else "An unrelated topic not described in the state."
    for question in result["questions"].values():
        if question["type"] != "choice":
            continue
        criteria = dict(question["criteria"])
        label, roll = question["label"], rng.random()
        none_key, distractor_key = "none_of_the_above", "unrelated_option"
        if len(criteria) > 2 and roll < p_none and none_key not in criteria:
            del criteria[label]
            criteria[none_key] = none_text
            question["label"] = none_key
        elif p_none <= roll < p_none + p_none_distract and len(criteria) < 255 and none_key not in criteria:
            criteria[none_key] = none_text
        elif p_none + p_none_distract <= roll < p_none + p_none_distract + p_distract and len(criteria) < 255 and distractor_key not in criteria:
            criteria[distractor_key] = distractor_text
        keys = list(criteria)
        rng.shuffle(keys)
        question["criteria"] = {key: criteria[key] for key in keys}
    return result


def none_pair(request: dict, rng: random.Random) -> list[dict]:
    """Same question twice: correct option present/absent, with none as distractor/answer."""
    eligible = [(qid, q) for qid, q in request["questions"].items()
                if q["type"] == "choice" and 3 <= len(q["criteria"]) < 255 and "none_of_the_above" not in q["criteria"]]
    if not eligible:
        return []
    qid, original = rng.choice(eligible)
    criterion = dict(original["criteria"])
    criterion["none_of_the_above"] = "None of the other options matches the state."
    keys = list(criterion)
    rng.shuffle(keys)
    present = {**copy.deepcopy(original), "criteria": {key: criterion[key] for key in keys}}
    absent = {**copy.deepcopy(present), "criteria": {key: value for key, value in present["criteria"].items() if key != original["label"]}, "label": "none_of_the_above"}
    result = []
    for variant, question in (("none_present", present), ("none_absent", absent)):
        meta = copy.deepcopy(request["_meta"])
        meta.update({"id": f"{meta['id']}/qev-{variant}", "variant": variant, "augmentation": "none-of-the-above minimal pair"})
        result.append({"state": copy.deepcopy(request["state"]), "questions": {qid: question}, "_meta": meta})
    return result


def augment_training(records: list[dict], seed: int, pair_fraction: float = 0.05) -> list[dict]:
    """Freeze augmentation once; all epochs see the same audited training examples."""
    result = []
    for request in records:
        rng = random.Random(f"augmentation:{seed}:{request['_meta']['id']}")
        changed = augment(request, rng)
        changed["_meta"]["augmentation"] = "fixed choice permutation and optional none/distractor"
        result.append(changed)
        if rng.random() < pair_fraction:
            result.extend(none_pair(request, rng))
    return result


def _require_split(split):
    if split not in SPLITS:
        raise ValueError("only train, calibration, and development are allowed; locked test is never read")


def _fetch_upstream(filename: str, cache_dir: Path) -> Path:
    if filename != "manifest.json":
        _require_split(Path(filename).stem)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(SUITE_REPOSITORY, f"{SUITE_PATH}/{filename}", repo_type="dataset",
                               revision=SUITE_REVISION, cache_dir=cache_dir))


def read_upstream(root: Path, split: str, manifest: dict, cache_dir: Path) -> tuple[list[dict], str]:
    _require_split(split)
    path = root / f"{split}.jsonl"
    if not path.exists():
        path = _fetch_upstream(f"{split}.jsonl", cache_dir)
    actual = digest(path)
    if actual != manifest["files"][f"{split}.jsonl"]["sha256"]:
        raise ValueError(f"upstream checksum mismatch: {split}")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != manifest["files"][f"{split}.jsonl"]["records"]:
        raise ValueError(f"upstream count mismatch: {split}")
    return records, actual


def select_public(records: list[dict], count_per_source: int, seed: int, split: str) -> list[dict]:
    """Choose clean original requests, balancing their original first-question labels."""
    _require_split(split)
    buckets = defaultdict(lambda: defaultdict(list))
    for record in records:
        meta = record["_meta"]
        if meta["source"] in PUBLIC_SOURCES and meta.get("variant", "clean") == "clean":
            first = next(iter(record["questions"].values()))
            buckets[meta["source"]][json.dumps(first["label"], sort_keys=True)].append(record)
    selected = []
    for source in PUBLIC_SOURCES:
        rng = random.Random(f"{seed}:{split}:{source}")
        labels = sorted(buckets[source])
        for values in buckets[source].values():
            values.sort(key=lambda row: row["_meta"]["id"])
            rng.shuffle(values)
        candidates = []
        while any(buckets[source].values()):
            rng.shuffle(labels)
            candidates.extend(buckets[source][label].pop() for label in labels if buckets[source][label])
        if len(candidates) < count_per_source:
            raise ValueError(f"not enough clean {source}/{split} rows: {len(candidates)} < {count_per_source}")
        for original in candidates[:count_per_source]:
            item = copy.deepcopy(original)
            item["_meta"]["qev_parent_partition"] = split
            item["_meta"]["language"] = "en"
            selected.append(item)
    return selected


def _rule_candidate(rng: random.Random, family: str, language: str) -> tuple[dict, str]:
    zh = language == "zh"
    if family == "refund":
        window, elapsed, unused = rng.randint(7, 90), rng.randint(0, 120), bool(rng.randrange(2))
        facts = {"family": family, "window": window, "elapsed": elapsed, "unused": unused}
        state = (f"规则：购买后不超过{window}天且商品未使用，才可退款。实际已过{elapsed}天，商品{'未使用' if unused else '已使用'}。"
                 if zh else f"A refund is allowed only within {window} days of purchase and when the item is unused. The purchase was {elapsed} days ago. The item is {'unused' if unused else 'used'}.")
        q = {"type": "noul", "instructions": "按规则可以退款吗？" if zh else "Is a refund permitted by the rule?", "label": elapsed <= window and unused}
    elif family == "approval":
        limit, amount, verified = rng.randint(1, 100) * 50, rng.randint(1, 150) * 50, bool(rng.randrange(2))
        facts = {"family": family, "limit": limit, "amount": amount, "verified": verified}
        state = (f"规则：身份已验证且金额不超过{limit}元时批准申请，否则拒绝。本次金额{amount}元，身份{'已验证' if verified else '未验证'}。"
                 if zh else f"Approve only if identity is verified and the amount is at most {limit}; otherwise reject. The amount is {amount}. Identity is {'verified' if verified else 'not verified'}.")
        q = {"type": "choice", "instructions": "根据规则选择处理结果。" if zh else "Choose the outcome under the rule.",
             "criteria": {"approve": "批准" if zh else "Approve", "reject": "拒绝" if zh else "Reject"},
             "label": "approve" if verified and amount <= limit else "reject"}
    elif family == "deadline":
        grace, late = rng.randint(1, 20), rng.randint(-20, 40)
        due = date(2025, 1, 1) + timedelta(days=rng.randint(0, 1600))
        received = due + timedelta(days=late)
        facts = {"family": family, "grace": grace, "late": late, "due": due.isoformat()}
        state = (f"规则：截止当天或之前提交为准时；逾期不超过{grace}天仍接受；更晚则拒绝。截止日期{due.isoformat()}，提交日期{received.isoformat()}。"
                 if zh else f"Submissions on or before the deadline are on time. Submissions at most {grace} days late are accepted late; later ones are rejected. Deadline: {due.isoformat()}. Submitted: {received.isoformat()}.")
        q = {"type": "score", "instructions": "属于哪个提交等级？" if zh else "Which submission level applies?",
             "criteria": ["准时", "逾期接受", "拒绝"] if zh else ["On time", "Late but accepted", "Rejected"],
             "label": 0 if late <= 0 else 1 if late <= grace else 2}
    else:
        low, width, spend = rng.randint(1, 30) * 100, rng.randint(1, 30) * 100, rng.randint(0, 70) * 100
        high = low + width
        facts = {"family": family, "low": low, "high": high, "spend": spend}
        state = (f"规则：消费低于{low}元为普通会员；达到{low}元但低于{high}元为银卡；达到{high}元为金卡。本期消费{spend}元。"
                 if zh else f"Spending below {low} gives standard membership; from {low} up to but excluding {high} gives silver; at least {high} gives gold. Spending this period is {spend}.")
        q = {"type": "score", "instructions": "会员等级是什么？" if zh else "What is the membership level?",
             "criteria": ["普通", "银卡", "金卡"] if zh else ["Standard", "Silver", "Gold"],
             "label": 0 if spend < low else 1 if spend < high else 2}
    semantic = canonical_hash(facts)
    q["src"] = f"synthetic_{family}_{language}"
    return {"state": state, "questions": {"decision": q},
            "_meta": {"source": f"synthetic_rules_{language}", "family": family, "language": language,
                      "semantic_sha256": semantic, "executable_facts": facts, "variant": "clean"}}, semantic


def generate_rules(count: int, split: str, seed: int, used_semantics: set[str] | None = None) -> list[dict]:
    """Same four task families, disjoint factual instances; not a novel-family test."""
    _require_split(split)
    used = used_semantics if used_semantics is not None else set()
    rng = random.Random(f"rules:{seed}:{split}")
    axes = [(family, lang) for family in ("refund", "approval", "deadline", "membership") for lang in ("en", "zh")]
    result = []
    for index in range(count):
        family, language = axes[index % len(axes)]
        for _ in range(10000):
            record, semantic = _rule_candidate(rng, family, language)
            if semantic not in used:
                break
        else:
            raise ValueError("unable to generate a unique rule instance")
        used.add(semantic)
        record["_meta"].update({"id": f"qev-rules/{semantic}", "group_id": f"qev-rules/{semantic}",
                                "qev_parent_partition": split, "license": "Apache-2.0",
                                "label_source": "executable rule, no model teacher"})
        result.append(record)
    return result


def verify_partitions(partitions: dict[str, list[dict]]) -> None:
    """Reject duplicate states/groups across partitions; validate every gold label."""
    seen_state, seen_group = {}, {}
    for split, records in partitions.items():
        _require_split(split)
        for request in records:
            materialize(request)
            state = " ".join(render(request["state"]).casefold().split())
            state_key = hashlib.sha256(state.encode()).hexdigest()
            group = request["_meta"].get("group_id", request["_meta"]["id"])
            for key, seen, label in ((state_key, seen_state, "state"), (group, seen_group, "group")):
                if key in seen and seen[key] != split:
                    raise ValueError(f"{label} overlap between {seen[key]} and {split}")
                seen[key] = split


def _license_provenance(repo: str, revision: str) -> dict:
    """Record the source's own Hub declaration, rather than inventing a license."""
    result = {"repository": repo, "revision": revision,
              "model_card_url": f"https://huggingface.co/datasets/{repo}/blob/{revision}/README.md"}
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(repo, revision=revision, token=False)
        card_data = info.card_data
        card = card_data.to_dict() if card_data is not None else {}
        result.update({"declared_license": card.get("license", "not_declared"),
                       "license_name": card.get("license_name"), "license_link": card.get("license_link"),
                       "card_metadata_sha256": canonical_hash(card)})
    except Exception as exc:  # noqa: BLE001 -- preserve unknown provenance on any metadata retrieval failure
        # Metadata retrieval is best effort; an unavailable declaration is
        # explicitly unknown, never silently replaced with the code's license.
        result.update({"declared_license": "not_verified", "metadata_error": str(exc)})
    return result


def prepare(out: str | Path, source_root: str | Path, train_per_source: int = 400,
            calibration_per_source: int = 40, development_per_source: int = 60,
            synthetic_train: int = 500, synthetic_calibration: int = 100, synthetic_development: int = 100,
            seed: int = 20260920, cache_dir: str | Path | None = None) -> dict:
    out, root = Path(out), Path(source_root)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {out}")
    counts = dict(zip(SPLITS, (train_per_source, calibration_per_source, development_per_source)))
    synth_counts = dict(zip(SPLITS, (synthetic_train, synthetic_calibration, synthetic_development)))
    if min(counts.values()) < 1 or min(synth_counts.values()) < 0:
        raise ValueError("public counts must be positive and synthetic counts nonnegative")
    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "qev" / "suites"
    parent_path = root / "manifest.json"
    if not parent_path.exists():
        parent_path = _fetch_upstream("manifest.json", cache)
    parent = json.loads(parent_path.read_text())
    partitions, input_hashes, used_semantics = {}, {}, set()
    for split in SPLITS:
        records, input_hashes[split] = read_upstream(root, split, parent, cache)
        partitions[split] = select_public(records, counts[split], seed, split)
        partitions[split] += generate_rules(synth_counts[split], split, seed, used_semantics)
    original_train_records = len(partitions["train"])
    partitions["train"] = augment_training(partitions["train"], seed)
    verify_partitions(partitions)
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(_license_provenance, repo, revision) for repo, revision in sorted(parent["dataset_revisions"].items())]
        licenses = [job.result() for job in jobs]
    manifest = {"format": "qev-data-v1", "seed": seed, "partitions": {},
                "upstream": {"repository": SUITE_REPOSITORY, "revision": SUITE_REVISION, "path": SUITE_PATH,
                             "manifest_sha256": digest(parent_path), "partition_sha256": input_hashes,
                             "conversion_code": UPSTREAM_CODE, "conversion_code_license": "Apache-2.0"},
                "source_datasets": licenses, "license": "Mixed source dataset licenses; see source_datasets",
                "synthetic": {"code": "qev/data.py", "code_sha256": digest(Path(__file__)), "license": "Apache-2.0",
                              "labels": "executable programmatic rules; no external model or Jev labels",
                              "languages": ["en", "zh"], "families": ["refund", "approval", "deadline", "membership"]},
                "protocol": {"locked_test_read": False, "original_partitions_preserved": True,
                             "training_augmentation": {"frozen_seed": seed, "original_records": original_train_records,
                                                       "choice_permutation": True, "p_none": 0.10,
                                                       "p_none_distractor": 0.12, "p_unrelated_distractor": 0.15,
                                                       "none_pair_fraction": 0.05,
                                                       "extra_pair_records": len(partitions["train"]) - original_train_records,
                                                       "calibration_and_development_augmented": False},
                             "development_selection": "clean original requests only; source and first-label balanced sampling",
                             "separation": "normalized exact states and group identities; synthetic semantic instances also disjoint",
                             "limitations": ["No fuzzy or pretraining-contamination audit", "Development participates in selection; not a final locked test",
                                              "Chinese evidence is restricted to four programmatic rule families", "Synthetic validation uses familiar families with new instances; no novel-family generalization claim"],
                             "calibration": "held aside for temperature fitting; softmax alone is not calibrated"}}
    out.mkdir(parents=True)
    for split, records in partitions.items():
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in records))
        types = Counter(q["type"] for row in records for q in row["questions"].values())
        manifest["partitions"][split] = {"sha256": digest(path), "records": len(records), "questions": sum(types.values()),
                                         "question_types": dict(types), "sources": dict(Counter(row["_meta"]["source"] for row in records))}
    write_json(out / "manifest.json", manifest)
    return manifest


def load_split(directory: str | Path, split: str) -> list[dict]:
    _require_split(split)
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = manifest["partitions"][split]
    path = root / f"{split}.jsonl"
    if digest(path) != expected["sha256"]:
        raise ValueError(f"dataset checksum mismatch: {split}")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != expected["records"]:
        raise ValueError(f"dataset count mismatch: {split}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--out", required=True)
    command.add_argument("--source-root", default="../kev/evals/public-pool-v4")
    command.add_argument("--train-per-source", type=int, default=400)
    command.add_argument("--calibration-per-source", type=int, default=40)
    command.add_argument("--development-per-source", type=int, default=60)
    command.add_argument("--synthetic-train", type=int, default=500)
    command.add_argument("--synthetic-calibration", type=int, default=100)
    command.add_argument("--synthetic-development", type=int, default=100)
    command.add_argument("--seed", type=int, default=20260920)
    command.add_argument("--cache-dir")
    args = vars(parser.parse_args())
    args.pop("command")
    result = prepare(**args)
    print(json.dumps({"partitions": result["partitions"], "locked_test_read": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
