"""Prepare or publish the verified v0.3.0 checkpoints without modifying weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, ModelCard

ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    "data_manifest.json", "evaluation.json", "initial_development.json",
    "provenance.json", "base_integrity.json", "export_integrity.json",
    "snake-native.json", "service_mlx.json", "parent-mlx-spatial.json",
    "trained-mlx-spatial.json", "parent-snake-torch.json", "trained-snake-torch.json",
    "parent-snake-torch-12.json", "trained-snake-torch-12.json",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release_files(flavor):
    suffix = "-mlx" if flavor == "mlx" else ""
    source = ROOT / "models" / f"qev-snake-0.8b{suffix}"
    files = {"qev_config.json": source / "qev_config.json",
             "pointer.safetensors": source / "pointer.safetensors"}
    folders = ("backbone",) if flavor == "mlx" else ("tokenizer", "processor")
    for folder in folders:
        for path in sorted((source / folder).rglob("*")):
            if path.is_file() and not path.is_symlink():
                files[str(path.relative_to(source))] = path
    if flavor == "mlx":
        files["decision_adapters.safetensors"] = source / "decision_adapters.safetensors"
        required = ("backbone/model.safetensors", "backbone/config.json",
                    "backbone/foundation_manifest.json", "backbone/tokenizer.json",
                    "backbone/processor_config.json", "backbone/LICENSE")
    else:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            files[f"adapter/{name}"] = source / "adapter" / name
        required = ("processor/processor_config.json", "processor/tokenizer.json",
                    "tokenizer/tokenizer.json")
    for name in required:
        if name not in files:
            raise FileNotFoundError(source / name)
    files["README.md"] = ROOT / "docs/huggingface" / f"qev-0.8b{suffix}.md"
    for name in ("LICENSE", "NOTICE"):
        files[name] = ROOT / name
    for name in REPORTS:
        files[f"reports/{name}"] = ROOT / "docs/results/snake_training" / name
    files["reports/general_training_data_manifest.json"] = ROOT / "docs/results/data_manifest.json"
    for path in files.values():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    config = json.loads(files["qev_config.json"].read_text())
    assert config["modalities"] == ["text", "image", "video"]
    assert config["native_generation"] == "adapter_disabled"
    assert config.get("runtime", "torch") == flavor
    assert config["base_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    ModelCard.load(files["README.md"])
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="twainsk")
    parser.add_argument("--flavor", choices=("torch", "mlx", "both"), default="both")
    parser.add_argument("--publish", action="store_true", help="Create public Hub repositories and upload")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/huggingface-publication")
    args = parser.parse_args()
    api = HfApi() if args.publish else None
    if api:
        account = api.whoami()
        namespaces = {account["name"], *(org["name"] for org in account.get("orgs", []))}
        if args.namespace not in namespaces:
            raise ValueError("Namespace is not the authenticated user or one of their organizations")
    args.output.mkdir(parents=True, exist_ok=True)
    for flavor in ("torch", "mlx") if args.flavor == "both" else (args.flavor,):
        name = "qev-0.8b" + ("-mlx" if flavor == "mlx" else "")
        repo_id = f"{args.namespace}/{name}"
        files = release_files(flavor)
        manifest = {
            "format": "qev-hub-release-v1", "version": "0.3.0", "repo_id": repo_id,
            "runtime": flavor, "runtime_source_commit": "d68c468d6c9963d558528df63c3c16dda711b84c",
            "files": {name: {"size": path.stat().st_size, "sha256": digest(path)}
                      for name, path in sorted(files.items())},
            "scope": "Runtime files and evaluation summaries; raw training records are not included.",
        }
        manifest_path = args.output / f"{name}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        files["release_manifest.json"] = manifest_path
        print(json.dumps({"repo": repo_id, "files": len(files),
                          "bytes": sum(p.stat().st_size for p in files.values()),
                          "publish": args.publish}), flush=True)
        if not api:
            continue
        # Existing non-empty releases are never overwritten by this initial publisher.
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        info = api.model_info(repo_id)
        existing = set(api.list_repo_files(repo_id)) - {".gitattributes"}
        if existing:
            raise FileExistsError(f"{repo_id} already contains model files; publish a reviewed update separately")
        commit = api.create_commit(
            repo_id=repo_id, repo_type="model", parent_commit=info.sha,
            commit_message="Publish Qev v0.3.0 with preserved multimodal foundation",
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                        for name, path in sorted(files.items())],
        )
        remote = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
        siblings = {file.rfilename: file for file in remote.siblings}
        expected_files = {**manifest["files"], "release_manifest.json": {
            "size": manifest_path.stat().st_size, "sha256": digest(manifest_path)}}
        for name, expected in expected_files.items():
            entry = siblings[name]
            assert entry.size == expected["size"], name
            if entry.lfs is not None:
                assert entry.lfs.sha256 == expected["sha256"], name
            else:
                data = files[name].read_bytes()
                blob = f"blob {len(data)}\0".encode() + data
                assert entry.blob_id == hashlib.sha1(blob).hexdigest(), name
        result = {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
                  "commit": commit.oid, "private": remote.private,
                  "verified_file_sizes": len(expected_files),
                  "verified_file_hashes": len(expected_files),
                  "verified_lfs_sha256": sum(siblings[name].lfs is not None for name in expected_files),
                  "manifest": str(manifest_path), "passed": True}
        (args.output / f"{repo_id.split('/')[-1]}.publication.json").write_text(
            json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
