"""Prepare or publish the v0.4.0 LFM2.5 decision checkpoints without modifying weights.

    uv run python scripts/publish_lfm2_models.py \
        --model qev-450m=runs/lfm2-eval/exports/qev-450m-a \
        --model qev-230m=runs/lfm2-eval/exports/qev-230m-a \
        --report qev-450m:evaluation.json=runs/qevd-results/vl-a/evaluation.json ... \
        [--publish]

Stages every repository file, writes the release manifest, and (with --publish)
creates the public Hub repository, uploads one commit and verifies the remote
file sizes and SHA-256 digests against the manifest. The packaged registry
manifest is written to qev/manifests/<name>.json either way; fill the printed
commit into qev/models.py after publishing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.4.0"
EXPECTED_MODALITIES = {"qev-450m": ["text", "image"], "qev-230m": ["text"]}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_model_cards(name, output):
    from huggingface_hub import ModelCard

    stem = f"{name}-mlx"
    source_navigation = f"[English]({stem}.md) | [简体中文]({stem}.zh-CN.md)"
    release_navigation = "[English](README.md) | [简体中文](README.zh-CN.md)"
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for language in ("", ".zh-CN"):
        source = ROOT / "docs/huggingface" / f"{stem}{language}.md"
        content = source.read_text()
        if content.count(source_navigation) != 1:
            raise ValueError(f"Expected one bilingual navigation in {source}")
        path = output / f"README{language}.md"
        path.write_text(content.replace(source_navigation, release_navigation))
        ModelCard.load(path)
        files[f"README{language}.md"] = path
    return files


def release_files(name, source, reports, card_output):
    source = Path(source)
    files = {}
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files[str(path.relative_to(source))] = path
    for required in ("qev_config.json", "pointer.safetensors", "decision_adapters.safetensors",
                     "backbone/model.safetensors", "backbone/config.json", "backbone/tokenizer.json",
                     "backbone/chat_template.jinja", "backbone/LICENSE", "LICENSE", "NOTICE"):
        if required not in files:
            raise FileNotFoundError(source / required)
    config = json.loads(files["qev_config.json"].read_text())
    assert config["format"] == "qev" and config["family"] == "lfm2" and config["runtime"] == "mlx"
    assert config["model_name"] == name and config["native_generation"] == "adapter_disabled"
    assert config["modalities"] == EXPECTED_MODALITIES[name]
    for report_name, path in reports.items():
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        json.loads(path.read_text())
        files[f"reports/{report_name}"] = path
    files.update(prepare_model_cards(name, card_output))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="twainsk")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=EXPORT_DIR")
    parser.add_argument("--report", action="append", default=[], metavar="NAME:FILE=PATH")
    parser.add_argument("--publish", action="store_true", help="Create public Hub repositories and upload")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/huggingface-publication")
    args = parser.parse_args()
    api = None
    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        account = api.whoami()
        namespaces = {account["name"], *(org["name"] for org in account.get("orgs", []))}
        if args.namespace not in namespaces:
            raise ValueError("Namespace is not the authenticated user or one of their organizations")
    commit_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                 capture_output=True, text=True).stdout.strip()
    for entry in args.model:
        name, source = entry.split("=", 1)
        repo_id = f"{args.namespace}/{name}-mlx"
        reports = {}
        for report in args.report:
            scope, path = report.split("=", 1)
            report_model, report_name = scope.split(":", 1)
            if report_model == name:
                reports[report_name] = path
        files = release_files(name, source, reports, args.output / f"{name}-mlx")
        manifest = {
            "format": "qev-hub-release-v1", "version": VERSION, "repo_id": repo_id, "runtime": "mlx",
            "runtime_source_commit": commit_hash,
            "files": {entry_name: {"size": path.stat().st_size, "sha256": digest(path)}
                      for entry_name, path in sorted(files.items())},
            "scope": "Runtime files and evaluation summaries; raw training records are not included.",
        }
        manifest_path = args.output / f"{name}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (ROOT / "qev/manifests" / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        files["release_manifest.json"] = manifest_path
        print(json.dumps({"repo": repo_id, "files": len(files),
                          "bytes": sum(path.stat().st_size for path in files.values()),
                          "publish": args.publish}), flush=True)
        if not api:
            continue
        from huggingface_hub import CommitOperationAdd

        # Existing non-empty releases are never overwritten by this initial publisher.
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        info = api.model_info(repo_id)
        existing = set(api.list_repo_files(repo_id)) - {".gitattributes"}
        if existing:
            raise FileExistsError(f"{repo_id} already contains model files; publish a reviewed update separately")
        commit = api.create_commit(
            repo_id=repo_id, repo_type="model", parent_commit=info.sha,
            commit_message=f"Publish Qev v{VERSION} {name} with unchanged LFM2.5 foundation",
            operations=[CommitOperationAdd(path_in_repo=entry_name, path_or_fileobj=str(path))
                        for entry_name, path in sorted(files.items())],
        )
        remote = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
        siblings = {file.rfilename: file for file in remote.siblings}
        expected = {**manifest["files"], "release_manifest.json": {
            "size": manifest_path.stat().st_size, "sha256": digest(manifest_path)}}
        problems = {}
        for entry_name, entry in expected.items():
            sibling = siblings.get(entry_name)
            checksum = getattr(getattr(sibling, "lfs", None), "sha256", None) if sibling else None
            if sibling is None:
                problems[entry_name] = "missing"
            elif sibling.size != entry["size"]:
                problems[entry_name] = f"size {sibling.size} != {entry['size']}"
            elif checksum is not None and checksum != entry["sha256"]:
                problems[entry_name] = "sha256 mismatch"
        if problems:
            raise RuntimeError(f"Remote verification failed for {repo_id}: {problems}")
        print(json.dumps({"repo": repo_id, "revision": commit.oid, "verified": len(expected)}), flush=True)


if __name__ == "__main__":
    main()
