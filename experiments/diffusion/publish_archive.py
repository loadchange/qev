"""Archive the six experiment checkpoints in a private Hugging Face repository.

Without ``--publish`` this only stages the cards and writes the manifest. The
upload refuses a non-empty repository and verifies every remote size and hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, ModelCard, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ("qwen35-causal", "qwen3-causal", "a2d-block", "qwen3-block", "lfm2-causal", "lfm2-block")
ARM_FILES = ("pointer.safetensors", "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
             "evaluation.json", "latency.json", "provenance.json", "snake.json", "training.jsonl",
             "development_rows.json", "train.log")
REPORTS = ("mm-probe.json", "order-probe.json", "paired-tests.json", "djev-zeroshot.json",
           "mlx-latency.json", "mlx-qev-adapters.json", "summary.json")
CODE_COMMIT = "664e6764d218d2e1a2baa4a40355a88d6d4d8501"
LFM = ("LiquidAI/LFM2.5-VL-450M", "fc6221ca597f3315e4f82fc2df606783267b34ba")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_files(runs, staging):
    files = {}
    for arm in ARMS:
        source = runs / arm
        names = list(ARM_FILES)
        if (source / "qevd_config.json").is_file():
            names.append("qevd_config.json")
        else:  # the Qwen3.5 arm uses the Qev checkpoint format with its tokenizer/processor
            names.append("qev_config.json")
            names += [str(p.relative_to(source)) for folder in ("tokenizer", "processor")
                      for p in sorted((source / folder).rglob("*")) if p.is_file()]
        for name in names:
            files[f"{arm}/{name}"] = source / name
        if arm.startswith("lfm2"):
            files[f"{arm}/LICENSE"] = staging / "LICENSE-LFM"
            files[f"{arm}/NOTICE"] = HERE / "huggingface/LFM-NOTICE"
    for name in REPORTS:
        files[f"reports/{name}"] = ROOT / "docs/results/diffusion" / name
    for name in ("LICENSE", "NOTICE"):
        files[name] = ROOT / name
    for name in ("README.md", "README.zh-CN.md", "LICENSES.md"):
        files[name] = HERE / "huggingface" / name
    missing = [str(path) for path in files.values() if not path.is_file() or path.is_symlink()]
    if missing:
        raise FileNotFoundError(missing)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="twainsk/qev-diffusion-experiment")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs/qevd-results")
    ap.add_argument("--output", type=Path, default=ROOT / "runs/huggingface-publication/qev-diffusion-experiment")
    ap.add_argument("--publish", action="store_true", help="Create the private repository and upload")
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download(LFM[0], "LICENSE", revision=LFM[1]), args.output / "LICENSE-LFM")
    files = archive_files(args.runs, args.output)
    for card in ("README.md", "README.zh-CN.md"):
        ModelCard.load(files[card])
    manifest = {"format": "qev-hub-archive-v1", "repo_id": args.repo_id, "private": True,
                "code_commit": CODE_COMMIT, "arms": list(ARMS),
                "files": {name: {"size": path.stat().st_size, "sha256": digest(path)}
                          for name, path in sorted(files.items())},
                "scope": "Experiment checkpoints (adapters, pointer heads) and reports; base weights not included."}
    manifest_path = args.output / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files["release_manifest.json"] = manifest_path
    print(json.dumps({"repo": args.repo_id, "files": len(files),
                      "bytes": sum(p.stat().st_size for p in files.values()), "publish": args.publish}))
    if not args.publish:
        return
    api = HfApi()
    if args.repo_id.split("/")[0] not in {api.whoami()["name"]}:
        raise ValueError("Repository namespace is not the authenticated user")
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(args.repo_id)
    if not info.private:
        raise RuntimeError(f"{args.repo_id} exists and is public; refusing to upload the archive there")
    if set(api.list_repo_files(args.repo_id)) - {".gitattributes"}:
        raise FileExistsError(f"{args.repo_id} already has files; upload a reviewed update separately")
    commit = api.create_commit(
        repo_id=args.repo_id, repo_type="model", parent_commit=info.sha,
        commit_message=f"Archive Qev diffusion experiment checkpoints (code {CODE_COMMIT[:7]})",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                    for name, path in sorted(files.items())])
    remote = api.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
    siblings = {file.rfilename: file for file in remote.siblings}
    expected = {**manifest["files"], "release_manifest.json": {"size": manifest_path.stat().st_size,
                                                                "sha256": digest(manifest_path)}}
    for name, entry in expected.items():
        sibling = siblings[name]
        assert sibling.size == entry["size"], name
        if sibling.lfs is not None:
            assert sibling.lfs.sha256 == entry["sha256"], name
        else:
            data = files[name].read_bytes()
            assert sibling.blob_id == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(), name
    result = {"repo_id": args.repo_id, "url": f"https://huggingface.co/{args.repo_id}", "commit": commit.oid,
              "private": remote.private, "code_commit": CODE_COMMIT, "verified_files": len(expected),
              "verified_lfs_sha256": sum(siblings[name].lfs is not None for name in expected), "passed": True}
    (args.output / "publication.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
