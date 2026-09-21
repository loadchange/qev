"""Refresh English and Chinese MODEL_CARD tables without loading a model.

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
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = (
    "provenance.json", "qev_config.json", "evaluation.json",
    "untrained_development.json", "base_integrity.json",
)


def translator(locale):
    if locale not in {"en", "zh-CN"}:
        raise ValueError(f"Unsupported model card locale: {locale}")
    return lambda english, chinese: english if locale == "en" else chinese


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


def status(value, locale="en"):
    tr = translator(locale)
    return (tr("Passed", "通过") if value is True else
            tr("Failed", "未通过") if value is False else tr("Pending validation", "待验证"))


def link(path, output):
    return f"[{path.name}]({os.path.relpath(path, output.parent)})"


def data_block(manifest, audit, manifest_path, audit_path, output, locale="en"):
    tr = translator(locale)
    pending, missing = tr("Pending validation", "待验证"), tr("Not recorded", "未记录")
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
                     found.get("max_tokens", pending), found.get("state_truncated", pending)])
    result = table([tr("Split", "分区"), "Requests", tr("Questions", "问题数"), "Choice", "Noul", "Score",
                    tr("Maximum tokens", "最大 token 数"), tr("State truncations", "State 截断数")], rows)
    aug = manifest.get("protocol", {}).get("training_augmentation", {})
    result += tr(
        f"\n\nData seed: `{manifest.get('seed', missing)}`. Before augmentation, training contained "
        f"{aug.get('original_records', missing)} requests; "
        f"{aug.get('extra_pair_records', missing)} paired none records were added. "
        f"Source: {link(manifest_path, output)}",
        f"\n\n数据种子：`{manifest.get('seed', missing)}`。训练增强前 "
        f"{aug.get('original_records', missing)} 条 request，新增 "
        f"{aug.get('extra_pair_records', missing)} 条 none 配对记录。"
        f"来源：{link(manifest_path, output)}")
    if audit:
        result += tr(f"; complete token audit: {link(audit_path, output)}",
                     f"；完整 token 审计：{link(audit_path, output)}")
    return result + tr(".", "。")


def training_block(core, paths, output, locale="en"):
    tr = translator(locale)
    missing = tr("Not recorded", "未记录")
    provenance, config = core.get("provenance.json"), core.get("qev_config.json") or {}
    if not provenance:
        return tr("Training provenance is unavailable for this run. The configuration above is planned; "
                  "completion, hardware and parameter counts remain pending artifact verification.",
                  "尚未读取到本次训练的 provenance；上文为确定的训练配置，完成状态、算力和参数量待产物核对。")
    args = provenance.get("arguments", {})
    evaluation = core.get("evaluation.json") or {}
    rows = [
        [tr("Completed epochs / configured epochs", "已完成 epoch / 配置 epoch"), f"{config.get('completed_epochs', missing)} / {args.get('epochs', missing)}"],
        [tr("Training device", "训练设备"), provenance.get("hardware", missing)],
        [tr("Training precision", "训练精度"), provenance.get("training_dtype", missing)],
        [tr("Trainable parameters", "可训练参数"), number(provenance.get("trainable_parameters"), 0)],
        [tr("Microbatch / gradient accumulation", "microbatch / 梯度累积"), f"{args.get('batch', missing)} / {args.get('accum', missing)}"],
        [tr("Initial learning rate / seed", "初始学习率 / seed"), f"{args.get('lr', missing)} / {args.get('seed', missing)}"],
        [tr("Training / calibration / development questions", "训练问题 / 校准问题 / 开发问题"), " / ".join(str(provenance.get("questions", {}).get(k, missing))
          for k in ("train", "calibration", "development"))],
        [tr("Optimizer updates", "optimizer 更新次数"), number(evaluation.get("optimizer_steps"), 0)],
        [tr("Training loop seconds, excluding evaluation", "训练循环耗时（秒，不含评估）"), number(evaluation.get("training_seconds"), 2)],
        [tr("Cumulative forward tokens, excluding padding", "累计前向 token（不含 padding）"), number(evaluation.get("forward_tokens"), 0)],
    ]
    versions = provenance.get("packages", {})
    result = table([tr("Item", "项目"), tr("Recorded value", "实际记录")], rows)
    result += tr("\n\nSoftware versions: ", "\n\n软件版本：") + ", ".join(
        f"`{k}=={v}`" for k, v in sorted(versions.items())) + tr(". ", "。")
    result += tr(f"Full arguments, source and data hashes: {link(paths['provenance.json'], output)}.",
                 f"完整参数、源码及数据摘要：{link(paths['provenance.json'], output)}。")
    limits = [f"{name}={args[name]}" for name in ("limit_train", "eval_limit") if args.get(name)]
    if limits:
        result += tr("\n\n**This run used subset limits: ", "\n\n**此次运行使用子集限制：") + ", ".join(limits) + tr(
            "; it must not be described as a full evaluation.**", "，不应称作全量评估。**")
    return result


def metric_row(label, values):
    return [label, values.get("n", "—"), percent(values.get("accuracy")),
            number(values.get("nll")), number(values.get("brier")),
            number(values.get("ece")), percent(values.get("confident_error_rate")),
            number(values.get("score_mae"))]


def evaluation_block(core, paths, output, results_dir, locale="en"):
    tr = translator(locale)
    missing = tr("Not recorded", "未记录")
    evaluation = core.get("evaluation.json")
    baseline = core.get("untrained_development.json")
    mlx_path = results_dir / "evaluation_mlx.json"
    mlx = read_json(mlx_path)
    rows = []
    if baseline:
        rows.append(metric_row(tr("Untrained random pointer · development", "训练前随机指针头 · development"), baseline["raw"]))
    if evaluation:
        dev = evaluation["development"]
        rows.extend([metric_row("CUDA BF16 · development · T=1", dev["raw"]),
                     metric_row(tr("CUDA BF16 · development · calibrated", "CUDA BF16 · development · 校准"), dev["calibrated"])])
    if mlx:
        mlx_dev = mlx["development"]
        if evaluation:
            if mlx_dev["raw"].get("n") != evaluation["development"]["raw"].get("n"):
                raise ValueError("CUDA and MLX development evaluation sizes differ")
            if not math.isclose(mlx_dev["temperature"], evaluation["development"]["temperature"],
                                rel_tol=0, abs_tol=1e-12):
                raise ValueError("MLX temperature differs from the original CUDA calibration")
        label = tr(f"Released MLX {mlx.get('mlx_dtype', missing)} · development",
                   f"交付 MLX {mlx.get('mlx_dtype', missing)} · development")
        rows.extend([metric_row(label + " · T=1", mlx_dev["raw"]),
                     metric_row(label + tr(" · reused calibration", " · 沿用校准"), mlx_dev["calibrated"])])
    if not rows:
        return tr("Pending validation: no pre-training baseline or post-training evaluation report is available for this checkpoint.",
                  "待验证：尚无本次 checkpoint 的训练前 baseline 或训练后 evaluation 报告。")
    metric_headers = [tr("Evaluation", "评估"), tr("Questions", "问题数"), tr("Accuracy", "准确率"),
                      "NLL", "Brier", "ECE", tr("High-confidence error rate", "高置信错误率"), "Score MAE"]
    result = table(metric_headers, rows)
    if baseline:
        result += tr(f"\n\nPre-training record: {link(paths['untrained_development.json'], output)}.",
                     f"\n\n训练前记录：{link(paths['untrained_development.json'], output)}。")
    if not evaluation:
        return result + tr(" Post-training results and temperature calibration remain pending validation.",
                           "训练后结果及温度校准待验证。")
    temp = evaluation["development"]["temperature"]
    if not isinstance(temp, (int, float)) or not math.isfinite(temp) or temp <= 0:
        raise ValueError("Evaluation temperature must be finite and positive")
    result += tr(f"\n\nTemperature `T={number(temp, 6)}`, fitted on "
                 f"`{evaluation.get('calibration_fit_partition', missing)}`. "
                 f"Report: {link(paths['evaluation.json'], output)}.",
                 f"\n\n温度 `T={number(temp, 6)}`，拟合分区："
                 f"`{evaluation.get('calibration_fit_partition', missing)}`。"
                 f"报告：{link(paths['evaluation.json'], output)}。")
    precision = evaluation.get("evaluation_precision", {})
    if precision:
        result += tr(f" Evaluation device: `{precision.get('device')}`; weights: `{precision.get('weight_dtype')}`; "
                     f"autocast: `{precision.get('autocast_dtype')}`; pointer: `{precision.get('pointer_dtype')}`.",
                     f"评估设备 `{precision.get('device')}`，权重 `{precision.get('weight_dtype')}`，"
                     f"autocast `{precision.get('autocast_dtype')}`，pointer `{precision.get('pointer_dtype')}`。")
    if mlx:
        result += tr(f"\n\nThe released MLX model uses per-question `{mlx.get('mlx_dtype', missing)}` inference "
                     f"and was evaluated on all {mlx['development']['raw']['n']} identical development questions: "
                     f"{link(mlx_path, output)}. MLX reuses the CUDA calibration temperature above, without "
                     "refitting on development or MLX outputs. CUDA BF16 batched evaluation and MLX FP32 "
                     "per-question evaluation differ in arithmetic precision, operators and batch shapes, "
                     "so metrics and some argmax choices can differ.",
                   f"\n\n交付 MLX 使用 `{mlx.get('mlx_dtype', missing)}` 逐题推理，"
                   f"完整评估 {mlx['development']['raw']['n']} 条相同 development 问题："
                   f"{link(mlx_path, output)}。MLX 沿用上述 CUDA calibration 温度，"
                   "没有在 development 或 MLX 输出上重新拟合。CUDA BF16 批量评估与 MLX FP32 "
                   "逐题评估的算术精度、算子和批量形状不同，指标及部分 argmax 可以不同。")
        comparison = mlx.get("cuda_bf16_comparison")
        if comparison:
            result += tr(f" After aligning question identities, {comparison.get('argmax_flips', missing)} "
                         f"of {comparison.get('questions', missing)} argmax choices changed; the maximum "
                         f"calibrated probability difference was {number(comparison.get('max_probability_difference'), 6)}, "
                         f"and the median per-question maximum difference was {number(comparison.get('median_probability_difference'), 6)}. "
                         "This is a cross-precision diagnostic, not the acceptance threshold for the "
                         "CPU FP32 → MLX FP32 export comparison below.",
                       f"逐条身份对齐后，在 {comparison.get('questions', missing)} 条问题中，"
                       f"argmax 改变 {comparison.get('argmax_flips', missing)} 条；"
                       f"校准概率最大差值 {number(comparison.get('max_probability_difference'), 6)}，"
                       f"每题最大差值的中位数 {number(comparison.get('median_probability_difference'), 6)}。"
                       "这是跨精度诊断，不等同于下文 CPU FP32 → MLX FP32 导出对照的验收门槛。")
    else:
        result += tr("\n\nFull development metrics for the released MLX model remain pending validation; "
                     "CUDA results do not substitute for MLX deployment results.",
                     "\n\n交付 MLX 的全量 development 指标待验证；CUDA 结果不直接充当 MLX 部署结果。")
    cal = evaluation.get("calibration", {})
    if cal:
        result += tr("\n\nCalibration fit split (used to select temperature, not independent evidence of quality):\n\n",
                     "\n\n校准拟合分区（用于选温度，不视作独立效果证据）：\n\n")
        result += table([tr("Split", "分区"), tr("Questions", "问题数"), "NLL · T=1", tr("NLL · calibrated", "NLL · 校准")], [[
            "calibration", cal["raw"].get("n"), number(cal["raw"].get("nll")),
            number(cal["calibrated"].get("nll"))]])
    for key, title in (("by_qtype", tr("Output type", "输出类型")), ("by_language", tr("Language", "语言")), ("by_source", tr("Data source", "数据来源"))):
        groups = evaluation["development"].get(key, {})
        if groups:
            result += tr(f"\n\nDevelopment by {title.lower()} (both paths apply the temperature above):\n\n",
                         f"\n\nDevelopment 按{title}划分（两条路径均应用上述温度）：\n\n") if mlx else tr(
                f"\n\nCUDA development by {title.lower()} (applying the temperature above):\n\n",
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
                headers = [title, tr("Questions", "问题数"), tr("CUDA accuracy", "CUDA 准确率"), tr("MLX accuracy", "MLX 准确率"), "CUDA NLL", "MLX NLL",
                           "CUDA ECE", "MLX ECE"]
                if key == "by_qtype":
                    headers.extend(["CUDA Score MAE", "MLX Score MAE"])
                result += table(headers, group_rows)
            else:
                result += table([title, *metric_headers[1:]],
                                [metric_row(name, values) for name, values in sorted(groups.items())])
    return result


def validation_block(core, paths, results_dir, mlx_checkpoint, output, locale="en"):
    tr = translator(locale)
    missing, pending = tr("Not recorded", "未记录"), tr("Pending validation", "待验证")
    result_status = lambda value: status(value, locale)
    rows, details = [], []
    integrity = core.get("base_integrity.json")
    if integrity:
        before, after = integrity.get("before"), integrity.get("after")
        consistent = bool(before and after and before == after)
        if integrity.get("unchanged") is True and not consistent:
            raise ValueError("Integrity claims unchanged but before/after records differ or are absent")
        rows.append([tr("Frozen foundation parameters", "冻结基座参数"), result_status(integrity.get("unchanged") is True and consistent),
                     link(paths["base_integrity.json"], output)])
        details.append(tr(f"Frozen parameters: {number((before or {}).get('parameters'), 0)}; "
                          f"SHA-256 before training: `{(before or {}).get('sha256', missing)}`; "
                          f"SHA-256 after training: `{(after or {}).get('sha256', missing)}`.",
                       f"冻结参数数：{number((before or {}).get('parameters'), 0)}；"
                       f"训练前 SHA-256：`{(before or {}).get('sha256', missing)}`；"
                       f"训练后 SHA-256：`{(after or {}).get('sha256', missing)}`。"))
    else:
        rows.append([tr("Frozen foundation parameters", "冻结基座参数"), pending,
                     tr("Missing base_integrity.json", "缺少 base_integrity.json")])
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
            rows.append([tr("Torch / MLX native generation token comparison", "Torch / MLX 原生生成 token 对照"),
                         result_status(equal == len(native)) + tr(f" ({equal}/{len(native)})", f"（{equal}/{len(native)}）"), link(path, output)])
            details.append(tr(f"Native generation examples in {link(path, output)}: ",
                              f"{link(path, output)} 的原生生成样例：")
                           + tr(", ", "、").join(f"`{name}`" for name in native)
                           + tr(". These compare greedy tokens for fixed inputs and do not evaluate multimodal accuracy.",
                                "。这是固定输入的贪心 token 对照，不是多模态准确率评估。"))
        if "max_probability_error" in report and "argmax_flips" in report:
            found_kinds.add("parity")
            signature = (report.get("encoded_cases_sha256", str(path)),
                         report.get("reference"), report.get("mlx_dtype"))
            if signature in seen_parity:
                continue
            seen_parity.add(signature)
            rows.append([tr("Torch / MLX candidate probabilities", "Torch / MLX 候选概率"), result_status(report.get("passed")), link(path, output)])
            details.append(tr(f"{link(path, output)}: {report.get('questions', missing)} questions, "
                              f"maximum raw probability error {number(report.get('max_probability_error'), 8)}, "
                              f"{report.get('argmax_flips', missing)} argmax changes, "
                              f"tolerance {number(report.get('atol'), 8)}, MLX dtype `{report.get('mlx_dtype', missing)}`.",
                           f"{link(path, output)}：{report.get('questions', missing)} 条问题，"
                           f"原始概率最大误差 {number(report.get('max_probability_error'), 8)}，"
                           f"argmax 改变 {report.get('argmax_flips', missing)}，"
                           f"容差 {number(report.get('atol'), 8)}，MLX dtype `{report.get('mlx_dtype', missing)}`。"))
            if "text_cases" in report and "media_cases" in report:
                details.append(tr(f"This includes {report['text_cases']} text and {report['media_cases']} multimodal questions; "
                                  f"reference path: `{report.get('reference', missing)}`.",
                               f"其中 {report['text_cases']} 条文字、{report['media_cases']} 条多模态；"
                               f"参考路径：`{report.get('reference', missing)}`。"))
            if report.get("calibrated"):
                cal = report["calibrated"]
                details.append(tr(f"The same report records maximum calibrated probability error {number(cal.get('max_probability_error'), 8)} "
                                  f"and {cal.get('argmax_flips', missing)} argmax changes.",
                               f"同一报告的校准概率最大误差 {number(cal.get('max_probability_error'), 8)}，"
                               f"argmax 改变 {cal.get('argmax_flips', missing)}。"))
        elif "checks" in report and isinstance(report["checks"], dict):
            found_kinds.add("service")
            checks = report["checks"]
            passed = sum(check.get("passed") is True for check in checks.values())
            rows.append([f"HTTP / SDK · {report.get('backend', path.stem)}",
                         result_status(report.get('passed')) + tr(f" ({passed}/{len(checks)})", f"（{passed}/{len(checks)}）"), link(path, output)])
        elif "image_native" in report and "video_native" in report:
            found_kinds.add("native")
            rows.append([tr("Native generation preservation after training", "训练后原生生成保留"), result_status(report.get("passed")), link(path, output)])
            details.append(tr(f"Image-fixture token comparison in {link(path, output)}: "
                              f"{result_status(report['image_native'].get('preservation_passed')).lower()}. "
                              "Video generation and visual pointer outputs are functional probes, not visual accuracy benchmarks.",
                           f"{link(path, output)} 的图像 fixture token 对照："
                           f"{result_status(report['image_native'].get('preservation_passed'))}。"
                           "视频生成与视觉指针输出为功能探针，不是视觉准确率基准。"))
        elif "permutation" in report and "independent_questions" in report:
            permutation = report["permutation"]
            rows.append([tr(f"Question isolation · {report.get('backend', path.stem)}", f"多题隔离 · {report.get('backend', path.stem)}"),
                         result_status(report["independent_questions"].get("passed")), link(path, output)])
            details.append(tr(f"{link(path, output)}: {percent(permutation.get('same_choice_fraction'))} of "
                              f"{permutation.get('questions', missing)} development questions retained their choice after candidate reversal.",
                           f"{link(path, output)}：{permutation.get('questions', missing)} 条开发问题"
                           f"候选反转后同选比例 {percent(permutation.get('same_choice_fraction'))}。"))
            observations = permutation.get("rows", [])
            if observations:
                same = sum(row.get("same_choice") is True for row in observations)
                worst = max(row["max_probability_difference"] for row in observations)
                changed = [row for row in observations if row.get("same_choice") is False]
                details.append(tr(f"Choices stayed the same for {same}/{len(observations)} questions; "
                                  f"the largest single-candidate probability change was {number(worst, 6)}. "
                                  "This probe does not support a guarantee of candidate-order invariance.",
                               f"其中 {same}/{len(observations)} 条保持同选，"
                               f"单候选概率最大变化 {number(worst, 6)}。"
                               "该探针不支持候选顺序不变性的保证。"))
                for row in changed[:5]:
                    details.append(tr(f"The flipped `{row.get('source')}` question had {row.get('options')} candidates: "
                                      f"the original choice was `{row.get('original_choice')}`, changing to `{row.get('reversed_choice')}` after reversal; "
                                      f"the largest single-candidate probability change was {number(row.get('max_probability_difference'), 6)}.",
                                   f"发生翻转的 `{row.get('source')}` 问题包含 {row.get('options')} 个候选，"
                                   f"原选择 `{row.get('original_choice')}`，反转后为 `{row.get('reversed_choice')}`，"
                                   f"单候选概率最大变化 {number(row.get('max_probability_difference'), 6)}。"))
            details.append(tr("Maximum probability difference between single-question, same-request multi-question and repeated requests: ",
                              "单题、同请求多题及重复请求之间的最大概率差值：")
                           + number(report["independent_questions"].get("maximum_probability_difference"), 8)
                           + tr(". This isolation check is separate from changing candidate order.",
                                "；此项隔离验证与改变候选顺序是不同测试。"))
        elif "load_seconds" in report and isinstance(report.get("results"), list):
            measurements = report["results"]
            details.append(tr(f"Warm sequential latency for fixed short requests ({link(path, output)}, "
                              f"backend `{report.get('backend')}`, {report.get('repeats', missing)} measurements for each question count):\n\n",
                           f"固定短请求的热启动顺序延迟（{link(path, output)}，"
                           f"backend `{report.get('backend')}`，每种问题数测量 "
                           f"{report.get('repeats', missing)} 次）：\n\n") + table(
                               [tr("Questions", "问题数"), tr("Total input tokens", "累计输入 token"), tr("Median ms", "中位耗时 ms"), tr("Questions/second", "问题/秒")],
                               [[row.get("questions"), row.get("input_tokens", missing), number(row.get("median_ms"), 2),
                                 number(row.get("questions_per_second"), 2)] for row in measurements]))
            details.append(tr(f"Latency platform: `{report.get('platform', missing)}`, architecture `{report.get('machine', missing)}`; "
                              f"model loading took {number(report.get('load_seconds'), 2)} seconds and is excluded from warm request latency.",
                           f"上述延迟平台：`{report.get('platform', missing)}`，"
                           f"架构 `{report.get('machine', missing)}`；"
                           f"模型加载耗时 {number(report.get('load_seconds'), 2)} 秒，不计入热请求延迟。"))
            hardware = report.get("hardware", {})
            if hardware:
                memory = hardware.get("memory_bytes")
                memory_label = number(memory / 1024 ** 3, 1) + " GiB" if memory is not None else missing
                details.append(tr(f"Chip: `{hardware.get('processor', missing)}`; unified memory: {memory_label}; "
                                  f"weight precision: `{report.get('dtype', missing)}`. "
                                  "Full software versions are recorded in the report's `packages` field.",
                               f"设备芯片：`{hardware.get('processor', missing)}`；"
                               f"统一内存：{memory_label}；权重精度：`{report.get('dtype', missing)}`。"
                               "软件版本完整记录于该报告的 `packages` 字段。"))
    for kind, label in (("native", tr("Post-training native generation comparison", "训练后原生生成对照")),
                        ("parity", tr("Post-training Torch / MLX comparison", "训练后 Torch / MLX 对照")),
                        ("service", tr("Actual checkpoint HTTP / SDK", "实际 checkpoint HTTP / SDK"))):
        if kind not in found_kinds:
            rows.append([label, pending, tr("No archived report", "尚无归档报告")])
    return table([tr("Validation", "验证"), tr("Status", "状态"), tr("Evidence", "证据")], rows) + ("\n\n" + "\n\n".join(details) if details else "")


def generated_blocks(manifest, audit, manifest_path, audit_path, core, paths,
                     results_dir, mlx_checkpoint, output, locale="en"):
    """Build both translations from identical evidence and validation logic."""
    tr = translator(locale)
    license_rows = []
    for item in manifest.get("source_datasets", []):
        license_value = item.get("declared_license", "not_declared")
        if isinstance(license_value, list):
            license_value = ", ".join(license_value)
        source_name = item["repository"]
        if item.get("model_card_url"):
            source_name = f"[{source_name}]({item['model_card_url']})"
        license_rows.append([source_name, license_value,
                             f"`{item.get('revision', tr('Not recorded', '未记录'))[:12]}`"])
    return {
        "DATA": data_block(manifest, audit, manifest_path, audit_path, output, locale),
        "TRAINING": training_block(core, paths, output, locale),
        "EVALUATION": evaluation_block(core, paths, output, results_dir, locale),
        "VALIDATION": validation_block(core, paths, results_dir, mlx_checkpoint, output, locale),
        "LICENSES": table([tr("Original data source", "原始数据来源"),
                           tr("License declared by source card", "来源卡片声明的许可"),
                           tr("Pinned revision (first 12 characters)", "固定 revision（前 12 位）")], license_rows),
    }


def document_template(content, template_path, output, outputs, locale):
    """Keep prose, with relative evidence links and same-language documentation."""
    translator(locale)  # Validate before changing links.
    content = re.sub(r"^\[English\]\([^\n]+\) \| \[简体中文\]\([^\n]+\)\n\n", "", content, count=1)

    def rewrite(match):
        destination = match.group(2)
        if destination.startswith(("#", "/")) or re.match(r"[a-zA-Z][\w+.-]*:", destination):
            return match.group(0)
        path, separator, fragment = destination.partition("#")
        target = (template_path.parent / path).resolve()
        if target.suffix == ".md":
            stem = target.stem.removesuffix(".zh-CN")
            localized = target.with_name(stem + (".zh-CN" if locale == "zh-CN" else "") + ".md")
            if localized.is_file():
                target = localized
        return match.group(1) + os.path.relpath(target, output.parent) + separator + fragment + ")"

    content = re.sub(r"(\[[^\]\n]*\]\()([^\s)]+)\)", rewrite, content)
    navigation = (f"[English]({os.path.relpath(outputs['en'], output.parent)}) | "
                  f"[简体中文]({os.path.relpath(outputs['zh-CN'], output.parent)})")
    return navigation + "\n\n" + content


def build_card(content, blocks, output):
    """Replace generated sections without changing handwritten conclusions."""
    for name, block in blocks.items():
        begin, end = f"<!-- BEGIN GENERATED {name} -->", f"<!-- END GENERATED {name} -->"
        if content.count(begin) != 1 or content.count(end) != 1:
            raise ValueError(f"Expected one {name} marker pair in {output}")
        start, finish = content.index(begin) + len(begin), content.index(end)
        if start >= finish:
            raise ValueError(f"Reversed {name} marker pair")
        content = content[:start] + "\n" + block + "\n" + content[finish:]
    return content


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/qev-0.8b")
    parser.add_argument("--mlx-checkpoint", type=Path, default=ROOT / "models/qev-0.8b-mlx")
    parser.add_argument("--results", type=Path, default=ROOT / "docs/results")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/MODEL_CARD.md",
                        help="English model card (existing prose is preserved)")
    parser.add_argument("--output-zh", type=Path,
                        help="Chinese model card (default: --output with .zh-CN before its suffix)")
    parser.add_argument("--archive-results", action="store_true")
    args = parser.parse_args(argv)
    outputs = {"en": args.output, "zh-CN": args.output_zh or args.output.with_name(
        args.output.stem + ".zh-CN" + args.output.suffix)}
    if outputs["en"].resolve() == outputs["zh-CN"].resolve():
        raise ValueError("English and Chinese model cards must have different output paths")
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
    documents = {}
    for locale, output in outputs.items():
        template = output if output.is_file() else ROOT / "docs" / (
            "MODEL_CARD.md" if locale == "en" else "MODEL_CARD.zh-CN.md")
        content = document_template(template.read_text(encoding="utf-8"), template,
                                    output, outputs, locale)
        blocks = generated_blocks(manifest, audit, manifest_path, audit_path, core, paths,
                                  args.results, args.mlx_checkpoint, output, locale)
        documents[locale] = build_card(content, blocks, output)
    # Check both templates and all evidence before writing either language.
    for locale, content in documents.items():
        outputs[locale].parent.mkdir(parents=True, exist_ok=True)
        outputs[locale].write_text(content, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "output_zh": str(outputs["zh-CN"]),
                      "artifact_source": str(source),
                      "available_training_reports": [name for name, value in core.items() if value],
                      "trained_evaluation_present": core.get("evaluation.json") is not None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
