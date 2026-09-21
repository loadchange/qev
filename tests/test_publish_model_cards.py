"""Check local publication preparation without model loading or Hub access."""

import hashlib
import json

import pytest
from huggingface_hub import ModelCard

from scripts import publish_models


def test_local_release_preparation_keeps_both_languages(tmp_path, monkeypatch):
    source_root = publish_models.ROOT
    staged_sources = {}
    for flavor in ("torch", "mlx"):
        suffix = "-mlx" if flavor == "mlx" else ""
        checkpoint = tmp_path / "models" / f"qev-snake-0.8b{suffix}"
        checkpoint.mkdir(parents=True)
        config = {
            "modalities": ["text", "image", "video"],
            "native_generation": "adapter_disabled",
            "runtime": flavor,
            "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        }
        (checkpoint / "qev_config.json").write_text(json.dumps(config))
        names = ["pointer.safetensors"]
        if flavor == "mlx":
            names += [
                "decision_adapters.safetensors", "backbone/model.safetensors",
                "backbone/config.json", "backbone/foundation_manifest.json",
                "backbone/tokenizer.json", "backbone/processor_config.json", "backbone/LICENSE",
            ]
        else:
            names += [
                "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                "processor/processor_config.json", "processor/tokenizer.json",
                "tokenizer/tokenizer.json",
            ]
        for name in names:
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"local fixture: never loaded as model weights")
        for language in ("", ".zh-CN"):
            name = f"qev-0.8b{suffix}{language}.md"
            original = (source_root / "docs/huggingface" / name).read_text()
            path = tmp_path / "docs/huggingface" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(original)
            staged_sources[path] = original
    for name in publish_models.REPORTS:
        path = tmp_path / "docs/results/snake_training" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    (tmp_path / "docs/results/data_manifest.json").write_text("{}\n")
    for name in ("LICENSE", "NOTICE"):
        (tmp_path / name).write_text("fixture\n")

    def no_hub_access(*args, **kwargs):
        pytest.fail("Local preparation must not create a Hub client")

    output = tmp_path / "prepared"
    monkeypatch.setattr(publish_models, "ROOT", tmp_path)
    monkeypatch.setattr(publish_models, "HfApi", no_hub_access)
    monkeypatch.setattr("sys.argv", ["publish_models.py", "--output", str(output)])
    publish_models.main()

    for stem in ("qev-0.8b", "qev-0.8b-mlx"):
        manifest = json.loads((output / f"{stem}.manifest.json").read_text())
        source_navigation = f"[English]({stem}.md) | [简体中文]({stem}.zh-CN.md)"
        release_navigation = "[English](README.md) | [简体中文](README.zh-CN.md)"
        for language in ("", ".zh-CN"):
            name = f"README{language}.md"
            prepared = output / stem / name
            source = tmp_path / "docs/huggingface" / f"{stem}{language}.md"
            assert prepared.read_text() == source.read_text().replace(
                source_navigation, release_navigation
            )
            assert ModelCard.load(prepared).data.to_dict() == ModelCard.load(source).data.to_dict()
            assert manifest["files"][name] == {
                "size": prepared.stat().st_size,
                "sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
            }
    assert all(path.read_text() == original for path, original in staged_sources.items())


@pytest.mark.parametrize("broken_card", ["missing", "navigation"])
def test_prepare_model_cards_requires_complete_navigation(tmp_path, monkeypatch, broken_card):
    source = tmp_path / "docs/huggingface"
    source.mkdir(parents=True)
    navigation = "[English](qev-0.8b.md) | [简体中文](qev-0.8b.zh-CN.md)"
    (source / "qev-0.8b.md").write_text(navigation)
    if broken_card == "navigation":
        (source / "qev-0.8b.zh-CN.md").write_text("# Missing navigation\n")
    monkeypatch.setattr(publish_models, "ROOT", tmp_path)
    output = tmp_path / "prepared"
    with pytest.raises(FileNotFoundError if broken_card == "missing" else ValueError):
        publish_models.prepare_model_cards("torch", output)
    assert not output.exists()
