"""Fill MODEL_CARD tables from real artifacts, without loading a model.

Missing artifacts remain pending. With --archive-results, copy training metadata
and MLX parity into docs/results so the evidence remains available without weights.
Never run this against synthetic fixtures when publishing the actual model card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = (
    "provenance.json", "qev_config.json", "evaluation.json",
    "untrained_development.json", "base_integrity.json",
)
PENDING = "待验证"


def read_json(path):
    if not path.is_file():
        return None
    result = json.loads(path.read_text(), parse_constant=lambda value: reject(value, path))
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return result


def reject(value, path):
    raise ValueError(f"Non-finite JSON value {value} in {path}")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(map(cell, headers)) + " |",
         "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    )


def number(value, digits=4):
    if value is None:
        return "—"
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Expected a finite numeric measurement, got {value!r}")
    return f"{value:,.{digits}f}"


def percent(value):
    return "—" if value is None else number(value * 100, 2) + "%"


def status(value):
    return "通过" if value is True else "未通过" if value is False else PENDING


def link(path, output):
    return f"[{path.name}]({os.path.relpath(path, output.parent)})"


def data_block(manifest, audit, manifest_path, audit_path, output):
    rows = []
    for split in ("train", "calibration", "development"):
        source = manifest["partitions"][split]
        found = (audit or {}).get(split, {})
        for key, audit_key in (("records", "requests"), ("questions", "questions")):
            if audit_key in found and source[key] != found[audit_key]:
                raise ValueError(f"Data audit and manifest disagree: {split}/{key}")
        qtypes = source["question_types"]
        rows.append([split, source["records"], source["questions"], qtypes.get("choice", 0),
                     qtypes.get("noul", 0), qtypes.get("score", 0),
                     found.get("max_tokens", PENDING), found.get("state_truncated", PENDING)])
    result = table(["分区", "Requests", "问题数", "Choice", "Noul", "Score",
                    "最大 token 数", "State 截断数"], rows)
    aug = manifest.get("protocol", {}).get("training_augmentation", {})
    result += (f"\n\n数据种子：`{manifest.get('seed', '未记录')}`。训练增强前 "
               f"{aug.get('original_records', '未记录')} 条 request，新增 "
               f"{aug.get('extra_pair_records', '未记录')} 条 none 配对记录。"
               f"来源：{link(manifest_path, output)}")
    if audit:
        result += f"；完整 token 审计：{link(audit_path, output)}"
    return result + "。"


def training_block(core, paths, output):
    provenance, config = core.get("provenance.json"), core.get("qev_config.json") or {}
    if not provenance:
        return "尚未读取到本次训练的 provenance；上文为确定的训练配置，完成状态、算力和参数量待产物核对。"
    args = provenance.get("arguments", {})
    evaluation = core.get("evaluation.json") or {}
    rows = [
        ["已完成 epoch / 配置 epoch", f"{config.get('completed_epochs', '未记录')} / {args.get('epochs', '未记录')}"],
        ["训练设备", provenance.get("hardware", "未记录")],
        ["训练精度", provenance.get("training_dtype", "未记录")],
        ["可训练参数", number(provenance.get("trainable_parameters"), 0)],
        ["microbatch / 梯度累积", f"{args.get('batch', '未记录')} / {args.get('accum', '未记录')}"],
        ["初始学习率 / seed", f"{args.get('lr', '未记录')} / {args.get('seed', '未记录')}"],
        ["训练问题 / 校准问题 / 开发问题", " / ".join(str(provenance.get("questions", {}).get(k, "未记录"))
          for k in ("train", "calibration", "development"))],
        ["optimizer 更新次数", number(evaluation.get("optimizer_steps"), 0)],
        ["训练循环耗时（秒，不含评估）", number(evaluation.get("training_seconds"), 2)],
        ["累计前向 token（不含 padding）", number(evaluation.get("forward_tokens"), 0)],
    ]
    versions = provenance.get("packages", {})
    result = table(["项目", "实际记录"], rows)
    result += "\n\n软件版本：" + ", ".join(f"`{k}=={v}`" for k, v in sorted(versions.items())) + "。"
    result += f"完整参数、源码及数据摘要：{link(paths['provenance.json'], output)}。"
    limits = [f"{name}={args[name]}" for name in ("limit_train", "eval_limit") if args.get(name)]
    if limits:
        result += "\n\n**此次运行使用子集限制：" + ", ".join(limits) + "，不应称作全量评估。**"
    return result


def metric_row(label, values):
    return [label, values.get("n", "—"), percent(values.get("accuracy")),
            number(values.get("nll")), number(values.get("brier")),
            number(values.get("ece")), percent(values.get("confident_error_rate")),
            number(values.get("score_mae"))]


def evaluation_block(core, paths, output, results_dir):
    evaluation = core.get("evaluation.json")
    baseline = core.get("untrained_development.json")
    mlx_path = results_dir / "evaluation_mlx.json"
    mlx = read_json(mlx_path)
    rows = []
    if baseline:
        rows.append(metric_row("训练前随机指针头 · development", baseline["raw"]))
    if evaluation:
        dev = evaluation["development"]
        rows.extend([metric_row("CUDA BF16 · development · T=1", dev["raw"]),
                     metric_row("CUDA BF16 · development · 校准", dev["calibrated"])])
    if mlx:
        mlx_dev = mlx["development"]
        if evaluation:
            if mlx_dev["raw"].get("n") != evaluation["development"]["raw"].get("n"):
                raise ValueError("CUDA and MLX development evaluation sizes differ")
            if not math.isclose(mlx_dev["temperature"], evaluation["development"]["temperature"],
                                rel_tol=0, abs_tol=1e-12):
                raise ValueError("MLX temperature differs from the original CUDA calibration")
        label = f"交付 MLX {mlx.get('mlx_dtype', '未记录')} · development"
        rows.extend([metric_row(label + " · T=1", mlx_dev["raw"]),
                     metric_row(label + " · 沿用校准", mlx_dev["calibrated"])])
    if not rows:
        return "待验证：尚无本次 checkpoint 的训练前 baseline 或训练后 evaluation 报告。"
    result = table(["评估", "问题数", "准确率", "NLL", "Brier", "ECE",
                    "高置信错误率", "Score MAE"], rows)
    if baseline:
        result += f"\n\n训练前记录：{link(paths['untrained_development.json'], output)}。"
    if not evaluation:
        return result + "训练后结果及温度校准待验证。"
    temp = evaluation["development"]["temperature"]
    if not isinstance(temp, (int, float)) or not math.isfinite(temp) or temp <= 0:
        raise ValueError("Evaluation temperature must be finite and positive")
    result += (f"\n\n温度 `T={number(temp, 6)}`，拟合分区："
               f"`{evaluation.get('calibration_fit_partition', '未记录')}`。"
               f"报告：{link(paths['evaluation.json'], output)}。")
    precision = evaluation.get("evaluation_precision", {})
    if precision:
        result += (f"评估设备 `{precision.get('device')}`，权重 `{precision.get('weight_dtype')}`，"
                   f"autocast `{precision.get('autocast_dtype')}`，pointer `{precision.get('pointer_dtype')}`。")
    if mlx:
        result += (f"\n\n交付 MLX 使用 `{mlx.get('mlx_dtype', '未记录')}` 逐题推理，"
                   f"完整评估 {mlx['development']['raw']['n']} 条相同 development 问题："
                   f"{link(mlx_path, output)}。MLX 沿用上述 CUDA calibration 温度，"
                   "没有在 development 或 MLX 输出上重新拟合。CUDA BF16 批量评估与 MLX FP32 "
                   "逐题评估的算术精度、算子和批量形状不同，指标及部分 argmax 可以不同。")
        comparison = mlx.get("cuda_bf16_comparison")
        if comparison:
            result += (f"逐条身份对齐后，在 {comparison.get('questions', '未记录')} 条问题中，"
                       f"argmax 改变 {comparison.get('argmax_flips', '未记录')} 条；"
                       f"校准概率最大差值 {number(comparison.get('max_probability_difference'), 6)}，"
                       f"每题最大差值的中位数 {number(comparison.get('median_probability_difference'), 6)}。"
                       "这是跨精度诊断，不等同于下文 CPU FP32 → MLX FP32 导出对照的验收门槛。")
    else:
        result += "\n\n交付 MLX 的全量 development 指标待验证；CUDA 结果不直接充当 MLX 部署结果。"
    cal = evaluation.get("calibration", {})
    if cal:
        result += "\n\n校准拟合分区（用于选温度，不视作独立效果证据）：\n\n"
        result += table(["分区", "问题数", "NLL · T=1", "NLL · 校准"], [[
            "calibration", cal["raw"].get("n"), number(cal["raw"].get("nll")),
            number(cal["calibrated"].get("nll"))]])
    for key, title in (("by_qtype", "输出类型"), ("by_language", "语言"), ("by_source", "数据来源")):
        groups = evaluation["development"].get(key, {})
        if groups:
            result += f"\n\nDevelopment 按{title}划分（两条路径均应用上述温度）：\n\n" if mlx else (
                f"\n\nCUDA development 按{title}划分（应用上述温度）：\n\n")
            if mlx:
                mlx_groups = mlx["development"].get(key, {})
                if set(mlx_groups) != set(groups):
                    raise ValueError(f"CUDA and MLX evaluation groups differ: {key}")
                group_rows = []
                for name, values in sorted(groups.items()):
                    observed = mlx_groups[name]
                    if values.get("n") != observed.get("n"):
                        raise ValueError(f"CUDA and MLX evaluation group sizes differ: {key}/{name}")
                    row = [name, values.get("n"), percent(values.get("accuracy")),
                           percent(observed.get("accuracy")), number(values.get("nll")),
                           number(observed.get("nll")), number(values.get("ece")),
                           number(observed.get("ece"))]
                    if key == "by_qtype":
                        row.extend([number(values.get("score_mae")), number(observed.get("score_mae"))])
                    group_rows.append(row)
                headers = [title, "问题数", "CUDA 准确率", "MLX 准确率", "CUDA NLL", "MLX NLL",
                           "CUDA ECE", "MLX ECE"]
                if key == "by_qtype":
                    headers.extend(["CUDA Score MAE", "MLX Score MAE"])
                result += table(headers, group_rows)
            else:
                result += table([title, "问题数", "准确率", "NLL", "Brier", "ECE",
                                 "高置信错误率", "Score MAE"],
                                [metric_row(name, values) for name, values in sorted(groups.items())])
    return result


def validation_block(core, paths, results_dir, mlx_checkpoint, output):
    rows, details = [], []
    integrity = core.get("base_integrity.json")
    if integrity:
        before, after = integrity.get("before"), integrity.get("after")
        consistent = bool(before and after and before == after)
        if integrity.get("unchanged") is True and not consistent:
            raise ValueError("Integrity claims unchanged but before/after records differ or are absent")
        rows.append(["冻结基座参数", status(integrity.get("unchanged") is True and consistent),
                     link(paths["base_integrity.json"], output)])
        details.append(f"冻结参数数：{number((before or {}).get('parameters'), 0)}；"
                       f"训练前 SHA-256：`{(before or {}).get('sha256', '未记录')}`；"
                       f"训练后 SHA-256：`{(after or {}).get('sha256', '未记录')}`。")
    else:
        rows.append(["冻结基座参数", PENDING, "缺少 base_integrity.json"])
    found_kinds, seen_parity = set(), set()
    candidates = sorted(results_dir.glob("*.json"))
    if not (results_dir / "mlx_parity.json").is_file() and (mlx_checkpoint / "parity.json").is_file():
        candidates.append(mlx_checkpoint / "parity.json")
    for path in candidates:
        if path.name in {*CORE_FILES, "data_manifest.json", "data_audit.json"} or path.name.endswith("_rows.json"):
            continue
        report = read_json(path)
        if not report or report.get("audit_only"):
            continue
        native = report.get("native_generation")
        if isinstance(native, dict) and native:
            equal = sum(value.get("tokens_equal") is True
                        and bool(value.get("torch_tokens"))
                        and value.get("torch_tokens") == value.get("mlx_tokens")
                        for value in native.values())
            rows.append(["Torch / MLX 原生生成 token 对照",
                         f"{status(equal == len(native))}（{equal}/{len(native)}）", link(path, output)])
            details.append(f"{link(path, output)} 的原生生成样例："
                           + "、".join(f"`{name}`" for name in native)
                           + "。这是固定输入的贪心 token 对照，不是多模态准确率评估。")
        if "max_probability_error" in report and "argmax_flips" in report:
            found_kinds.add("parity")
            signature = (report.get("encoded_cases_sha256", str(path)),
                         report.get("reference"), report.get("mlx_dtype"))
            if signature in seen_parity:
                continue
            seen_parity.add(signature)
            rows.append(["Torch / MLX 候选概率", status(report.get("passed")), link(path, output)])
            details.append(f"{link(path, output)}：{report.get('questions', '未记录')} 条问题，"
                           f"原始概率最大误差 {number(report.get('max_probability_error'), 8)}，"
                           f"argmax 改变 {report.get('argmax_flips', '未记录')}，"
                           f"容差 {number(report.get('atol'), 8)}，MLX dtype `{report.get('mlx_dtype', '未记录')}`。")
            if "text_cases" in report and "media_cases" in report:
                details.append(f"其中 {report['text_cases']} 条文字、{report['media_cases']} 条多模态；"
                               f"参考路径：`{report.get('reference', '未记录')}`。")
            if report.get("calibrated"):
                cal = report["calibrated"]
                details.append(f"同一报告的校准概率最大误差 {number(cal.get('max_probability_error'), 8)}，"
                               f"argmax 改变 {cal.get('argmax_flips', '未记录')}。")
        elif "checks" in report and isinstance(report["checks"], dict):
            found_kinds.add("service")
            checks = report["checks"]
            passed = sum(check.get("passed") is True for check in checks.values())
            rows.append([f"HTTP / SDK · {report.get('backend', path.stem)}",
                         f"{status(report.get('passed'))}（{passed}/{len(checks)}）", link(path, output)])
        elif "image_native" in report and "video_native" in report:
            found_kinds.add("native")
            rows.append(["训练后原生生成保留", status(report.get("passed")), link(path, output)])
            details.append(f"{link(path, output)} 的图像 fixture token 对照："
                           f"{status(report['image_native'].get('preservation_passed'))}。"
                           "视频生成与视觉指针输出为功能探针，不是视觉准确率基准。")
        elif "permutation" in report and "independent_questions" in report:
            permutation = report["permutation"]
            rows.append([f"多题隔离 · {report.get('backend', path.stem)}",
                         status(report["independent_questions"].get("passed")), link(path, output)])
            details.append(f"{link(path, output)}：{permutation.get('questions', '未记录')} 条开发问题"
                           f"候选反转后同选比例 {percent(permutation.get('same_choice_fraction'))}。")
            observations = permutation.get("rows", [])
            if observations:
                same = sum(row.get("same_choice") is True for row in observations)
                worst = max(row["max_probability_difference"] for row in observations)
                changed = [row for row in observations if row.get("same_choice") is False]
                details.append(f"其中 {same}/{len(observations)} 条保持同选，"
                               f"单候选概率最大变化 {number(worst, 6)}。"
                               "该探针不支持候选顺序不变性的保证。")
                for row in changed[:5]:
                    details.append(f"发生翻转的 `{row.get('source')}` 问题包含 {row.get('options')} 个候选，"
                                   f"原选择 `{row.get('original_choice')}`，反转后为 `{row.get('reversed_choice')}`，"
                                   f"单候选概率最大变化 {number(row.get('max_probability_difference'), 6)}。")
            details.append("单题、同请求多题及重复请求之间的最大概率差值："
                           + number(report["independent_questions"].get("maximum_probability_difference"), 8)
                           + "；此项隔离验证与改变候选顺序是不同测试。")
        elif "load_seconds" in report and isinstance(report.get("results"), list):
            measurements = report["results"]
            details.append(f"固定短请求的热启动顺序延迟（{link(path, output)}，"
                           f"backend `{report.get('backend')}`，每种问题数测量 "
                           f"{report.get('repeats', '未记录')} 次）：\n\n" + table(
                               ["问题数", "累计输入 token", "中位耗时 ms", "问题/秒"],
                               [[row.get("questions"), row.get("input_tokens", "未记录"), number(row.get("median_ms"), 2),
                                 number(row.get("questions_per_second"), 2)] for row in measurements]))
            details.append(f"上述延迟平台：`{report.get('platform', '未记录')}`，"
                           f"架构 `{report.get('machine', '未记录')}`；"
                           f"模型加载耗时 {number(report.get('load_seconds'), 2)} 秒，不计入热请求延迟。")
            hardware = report.get("hardware", {})
            if hardware:
                memory = hardware.get("memory_bytes")
                memory_label = number(memory / 1024 ** 3, 1) + " GiB" if memory is not None else "未记录"
                details.append(f"设备芯片：`{hardware.get('processor', '未记录')}`；"
                               f"统一内存：{memory_label}；权重精度：`{report.get('dtype', '未记录')}`。"
                               "软件版本完整记录于该报告的 `packages` 字段。")
    for kind, label in (("native", "训练后原生生成对照"), ("parity", "训练后 Torch / MLX 对照"),
                        ("service", "实际 checkpoint HTTP / SDK")):
        if kind not in found_kinds:
            rows.append([label, PENDING, "尚无归档报告"])
    return table(["验证", "状态", "证据"], rows) + ("\n\n" + "\n\n".join(details) if details else "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/qev-0.8b")
    parser.add_argument("--mlx-checkpoint", type=Path, default=ROOT / "models/qev-0.8b-mlx")
    parser.add_argument("--results", type=Path, default=ROOT / "docs/results")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/MODEL_CARD.md")
    parser.add_argument("--archive-results", action="store_true")
    args = parser.parse_args(argv)
    # Do not fill gaps in a partial current checkpoint from stale archived runs.
    source = args.checkpoint if args.checkpoint.is_dir() else args.results
    paths = {name: source / name for name in CORE_FILES}
    core = {name: read_json(path) for name, path in paths.items()}
    manifest_path, audit_path = args.results / "data_manifest.json", args.results / "data_audit.json"
    manifest, audit = read_json(manifest_path), read_json(audit_path)
    if not manifest:
        raise FileNotFoundError(f"Required data manifest: {manifest_path}")
    provenance = core.get("provenance.json") or {}
    expected = provenance.get("dataset_manifest_sha256")
    if expected and hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Training provenance does not match the archived data manifest bytes")
    if args.archive_results:
        args.results.mkdir(parents=True, exist_ok=True)
        for name, path in paths.items():
            destination = args.results / name
            if path.is_file() and path.resolve() != destination.resolve():
                shutil.copy2(path, destination)
                paths[name] = destination
        parity = args.mlx_checkpoint / "parity.json"
        if parity.is_file():
            shutil.copy2(parity, args.results / "mlx_parity.json")
    license_rows = []
    for item in manifest.get("source_datasets", []):
        license_value = item.get("declared_license", "not_declared")
        if isinstance(license_value, list):
            license_value = ", ".join(license_value)
        source_name = item["repository"]
        if item.get("model_card_url"):
            source_name = f"[{source_name}]({item['model_card_url']})"
        license_rows.append([source_name, license_value, f"`{item.get('revision', '未记录')[:12]}`"])
    blocks = {
        "DATA": data_block(manifest, audit, manifest_path, audit_path, args.output),
        "TRAINING": training_block(core, paths, args.output),
        "EVALUATION": evaluation_block(core, paths, args.output, args.results),
        "VALIDATION": validation_block(core, paths, args.results, args.mlx_checkpoint, args.output),
        "LICENSES": table(["原始数据来源", "来源卡片声明的许可", "固定 revision（前 12 位）"], license_rows),
    }
    content = args.output.read_text()
    for name, block in blocks.items():
        begin, end = f"<!-- BEGIN GENERATED {name} -->", f"<!-- END GENERATED {name} -->"
        if content.count(begin) != 1 or content.count(end) != 1:
            raise ValueError(f"Expected one {name} marker pair in {args.output}")
        start, finish = content.index(begin) + len(begin), content.index(end)
        if start >= finish:
            raise ValueError(f"Reversed {name} marker pair")
        content = content[:start] + "\n" + block + "\n" + content[finish:]
    args.output.write_text(content)
    print(json.dumps({"output": str(args.output), "artifact_source": str(source),
                      "available_training_reports": [name for name, value in core.items() if value],
                      "trained_evaluation_present": core.get("evaluation.json") is not None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
