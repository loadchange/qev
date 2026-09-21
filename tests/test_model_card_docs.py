"""Model-card generation only reads metadata; no weights or inference are involved."""
import importlib.util
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("update_model_card_docs", ROOT / "scripts/update_model_card.py")
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)


def generated(content, name):
    return content.split(f"<!-- BEGIN GENERATED {name} -->", 1)[1].split(
        f"<!-- END GENERATED {name} -->", 1)[0].strip()


def arguments(tmp_path):
    return ["--checkpoint", str(tmp_path / "absent-checkpoint"),
            "--mlx-checkpoint", str(tmp_path / "absent-mlx-checkpoint"),
            "--results", str(ROOT / "docs/results"),
            "--output", str(tmp_path / "MODEL_CARD.md")]


def test_archived_evidence_generates_both_languages_without_changing_facts(tmp_path, capsys):
    updater.main(arguments(tmp_path))
    english = (tmp_path / "MODEL_CARD.md").read_text()
    chinese = (tmp_path / "MODEL_CARD.zh-CN.md").read_text()
    summary = json.loads(capsys.readouterr().out)
    assert summary["output_zh"] == str(tmp_path / "MODEL_CARD.zh-CN.md")
    assert summary["artifact_source"] == str(ROOT / "docs/results")
    assert summary["trained_evaluation_present"] is True
    assert english.startswith("[English](MODEL_CARD.md) | [简体中文](MODEL_CARD.zh-CN.md)")
    assert chinese.startswith("[English](MODEL_CARD.md) | [简体中文](MODEL_CARD.zh-CN.md)")
    assert "| Completed epochs / configured epochs | 2 / 2 |" in english
    assert "| 已完成 epoch / 配置 epoch | 2 / 2 |" in chinese
    for name in ("DATA", "TRAINING", "EVALUATION", "VALIDATION", "LICENSES"):
        en_block, zh_block = generated(english, name), generated(chinese, name)
        assert not re.search(r"[\u4e00-\u9fff]", en_block), name
        # Code literals and numeric table cells preserve actual evidence across translations.
        assert Counter(re.findall(r"`([^`]+)`", en_block)) == Counter(re.findall(r"`([^`]+)`", zh_block))
        numeric_cells = r"\| ([\d.,/% —]+) (?=\|)"
        assert Counter(re.findall(numeric_cells, en_block)) == Counter(re.findall(numeric_cells, zh_block))
        assert Counter(re.findall(r"\]\(([^)]+)\)", en_block)) == Counter(re.findall(r"\]\(([^)]+)\)", zh_block))
    for locale, content in (("en", english), ("zh-CN", chinese)):
        for destination in re.findall(r"\]\(([^)]+)\)", content):
            if "://" in destination or destination.startswith("#"):
                continue
            target = tmp_path / destination.split("#", 1)[0]
            assert target.is_file(), destination
            if target.suffix == ".md" and target.name not in {"MODEL_CARD.md", "MODEL_CARD.zh-CN.md"}:
                assert (".zh-CN." in target.name) == (locale == "zh-CN"), destination
    # Updating again keeps navigation, prose and links stable.
    updater.main(arguments(tmp_path))
    assert (tmp_path / "MODEL_CARD.md").read_text() == english
    assert (tmp_path / "MODEL_CARD.zh-CN.md").read_text() == chinese


def test_default_cli_archives_and_refreshes_both_cards(tmp_path, monkeypatch):
    docs, checkpoint = tmp_path / "docs", tmp_path / "models/qev-0.8b"
    docs.mkdir(); checkpoint.mkdir(parents=True)
    shutil.copytree(ROOT / "docs/results", docs / "results")
    for name in ("MODEL_CARD.md", "MODEL_CARD.zh-CN.md"):
        shutil.copy2(ROOT / "docs" / name, docs / name)
    for name in updater.CORE_FILES:
        shutil.copy2(ROOT / "docs/results" / name, checkpoint / name)
    monkeypatch.setattr(updater, "ROOT", tmp_path)
    updater.main(["--archive-results"])
    assert "| Split | Requests | Questions |" in (docs / "MODEL_CARD.md").read_text()
    assert "| 分区 | Requests | 问题数 |" in (docs / "MODEL_CARD.zh-CN.md").read_text()
    for name in updater.CORE_FILES:
        assert (docs / "results" / name).read_bytes() == (checkpoint / name).read_bytes()


def test_partial_current_checkpoint_does_not_fill_training_gaps_from_archive(tmp_path):
    checkpoint = tmp_path / "partial-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "qev_config.json").write_text('{"completed_epochs": 1}')
    args = arguments(tmp_path)
    args[1] = str(checkpoint)
    updater.main(args)
    english = (tmp_path / "MODEL_CARD.md").read_text()
    chinese = (tmp_path / "MODEL_CARD.zh-CN.md").read_text()
    assert "Training provenance is unavailable" in generated(english, "TRAINING")
    assert "尚未读取到本次训练的 provenance" in generated(chinese, "TRAINING")
    assert "Post-training results and temperature calibration remain pending validation" in generated(english, "EVALUATION")
    assert "训练后结果及温度校准待验证" in generated(chinese, "EVALUATION")
    assert "Frozen foundation parameters | Pending validation" in generated(english, "VALIDATION")
    assert "冻结基座参数 | 待验证" in generated(chinese, "VALIDATION")


def test_invalid_second_template_does_not_write_first_card(tmp_path):
    first = tmp_path / "MODEL_CARD.md"
    second = tmp_path / "MODEL_CARD.zh-CN.md"
    original = (ROOT / "docs/MODEL_CARD.md").read_text()
    first.write_text(original)
    second.write_text("Missing generated sections")
    with pytest.raises(ValueError, match="marker pair"):
        updater.main(arguments(tmp_path))
    assert first.read_text() == original
    assert second.read_text() == "Missing generated sections"


def test_custom_chinese_destination_preserves_its_prose_and_relative_links(tmp_path):
    english, chinese = tmp_path / "en/card.md", tmp_path / "zh/card.md"
    chinese.parent.mkdir()
    chinese.write_text((ROOT / "docs/MODEL_CARD.zh-CN.md").read_text() + "\n手写备注应保留。\n")
    args = arguments(tmp_path)
    args[-1] = str(english)
    updater.main([*args, "--output-zh", str(chinese)])
    assert chinese.read_text().endswith("手写备注应保留。\n")
    assert english.read_text().startswith("[English](card.md) | [简体中文](../zh/card.md)")
    assert chinese.read_text().startswith("[English](../en/card.md) | [简体中文](card.md)")


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
def test_evidence_consistency_checks_apply_to_both_languages(tmp_path, locale):
    results = ROOT / "docs/results"
    manifest = updater.read_json(results / "data_manifest.json")
    audit = updater.read_json(results / "data_audit.json")
    audit["train"]["requests"] += 1
    with pytest.raises(ValueError, match="audit and manifest disagree"):
        updater.data_block(manifest, audit, results / "data_manifest.json", results / "data_audit.json", tmp_path / "card.md", locale)
    paths = {name: results / name for name in updater.CORE_FILES}
    core = {name: updater.read_json(path) for name, path in paths.items()}
    core["evaluation.json"]["development"]["temperature"] += 0.1
    with pytest.raises(ValueError, match="temperature differs"):
        updater.evaluation_block(core, paths, tmp_path / "card.md", results, locale)
    core["base_integrity.json"]["after"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="Integrity claims unchanged"):
        updater.validation_block(core, paths, results, tmp_path / "missing", tmp_path / "card.md", locale)
