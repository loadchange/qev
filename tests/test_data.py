import json
import random

import pytest

from qev.data import (
    PUBLIC_SOURCES,
    SPLITS,
    augment,
    digest,
    generate_rules,
    load_split,
    materialize,
    none_pair,
    prepare,
    select_public,
    verify_partitions,
)


def request(identifier="one", state="A", source="agnews", label="a"):
    return {"state": state, "questions": {"topic": {"type": "choice", "instructions": "Which?",
            "criteria": {"a": "One", "b": "Two", "c": "Three"}, "label": label, "src": source}},
            "_meta": {"id": identifier, "group_id": identifier, "source": source, "variant": "clean"}}


def test_labels_are_attached_after_serving_render_and_augmentation_is_correct():
    original = request()
    augmented = augment(original, random.Random(1), p_none=1, p_none_distract=0, p_distract=0)
    assert original["questions"]["topic"]["label"] == "a"
    question = augmented["questions"]["topic"]
    assert "a" not in question["criteria"]
    assert question["label"] == "none_of_the_above"
    encoded = materialize(augmented)["questions"][0]
    assert encoded["keys"][encoded["label"]] == "none_of_the_above"
    assert "label" not in encoded["instr"]


def test_invalid_gold_label_rejected():
    row = request(label="nonexistent")
    with pytest.raises(ValueError):
        materialize(row)


def test_none_pair_preserves_order_and_changes_only_true_option_presence():
    present, absent = none_pair(request(), random.Random(3))
    a, b = present["questions"]["topic"], absent["questions"]["topic"]
    assert present["state"] == absent["state"]
    assert present["_meta"]["group_id"] == absent["_meta"]["group_id"]
    assert list(b["criteria"]) == [key for key in a["criteria"] if key != "a"]
    assert a["label"] == "a" and b["label"] == "none_of_the_above"


def test_disjoint_executable_bilingual_rules_and_deterministic_generation():
    used = set()
    partitions = {split: generate_rules(80, split, 7, used) for split in SPLITS}
    verify_partitions(partitions)
    assert len(used) == 240
    assert partitions["train"] == generate_rules(80, "train", 7)
    for rows in partitions.values():
        assert {row["_meta"]["language"] for row in rows} == {"en", "zh"}
        for row in rows:
            facts = row["_meta"]["executable_facts"]
            q = row["questions"]["decision"]
            family = facts["family"]
            if family == "refund":
                assert q["label"] == (facts["elapsed"] <= facts["window"] and facts["unused"])
            elif family == "approval":
                assert q["label"] == ("approve" if facts["verified"] and facts["amount"] <= facts["limit"] else "reject")
            elif family == "deadline":
                assert q["label"] == (0 if facts["late"] <= 0 else 1 if facts["late"] <= facts["grace"] else 2)
            else:
                assert q["label"] == (0 if facts["spend"] < facts["low"] else 1 if facts["spend"] < facts["high"] else 2)


def test_cross_partition_state_and_group_leaks_rejected():
    with pytest.raises(ValueError, match="state overlap"):
        verify_partitions({"train": [request()], "development": [request("two", " a ")]})
    with pytest.raises(ValueError, match="group overlap"):
        verify_partitions({"train": [request()], "development": [request(state="B")]})


def test_selection_never_promotes_development_or_augmented_variants_to_train():
    records = []
    for source in PUBLIC_SOURCES:
        records.extend(request(f"{source}-{i}", f"{source} {i}", source, "abc"[i % 3]) for i in range(9))
        variant = request(f"{source}-bad", f"{source}-bad", source)
        variant["_meta"]["variant"] = "none_absent"
        records.append(variant)
    selected = select_public(records, 3, 7, "development")
    assert len(selected) == 30
    assert all(row["_meta"]["variant"] == "clean" and row["_meta"]["qev_parent_partition"] == "development" for row in selected)


def test_prepare_preserves_splits_hashes_and_never_opens_locked_test(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    manifest = {"files": {}, "dataset_revisions": {}}
    for split in SPLITS:
        rows = [request(f"{split}-{source}-{i}", f"{split} {source} {i}", source, "abc"[i]) for source in PUBLIC_SOURCES for i in range(3)]
        path = parent / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(rows)}
    (parent / "manifest.json").write_text(json.dumps(manifest))
    # A locked file exists, but any attempt to read it causes the test to fail.
    locked = parent / "test.jsonl"
    locked.write_text("must never read")
    original_read = type(locked).read_text
    def guarded_read(path, *args, **kwargs):
        assert path != locked, "locked test was opened"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(type(locked), "read_text", guarded_read)
    out = tmp_path / "prepared"
    result = prepare(out, parent, 2, 1, 1, 8, 8, 8)
    assert result["protocol"]["locked_test_read"] is False
    assert result["partitions"]["train"]["records"] == 28 + result["protocol"]["training_augmentation"]["extra_pair_records"]
    assert result["partitions"]["development"]["records"] == 18
    loaded = {split: load_split(out, split) for split in SPLITS}
    verify_partitions(loaded)
    with pytest.raises(ValueError, match="locked test"):
        load_split(out, "test")
    with pytest.raises(FileExistsError):
        prepare(out, parent)
    with (out / "train.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_split(out, "train")
