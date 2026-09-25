"""Summarize every arm into one markdown table plus a JSON digest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = {  # label: (backbone, attention, question answered)
    "qwen35-causal": ("Qwen3.5-0.8B (current Qev)", "causal", "baseline, same recipe"),
    "qwen3-causal": ("Qwen3-0.6B autoregressive", "causal", "control"),
    "a2d-block": ("Qwen3-0.6B diffusion (dLLM MDLM)", "block", "diffusion model"),
    "qwen3-block": ("Qwen3-0.6B autoregressive", "block", "bidirectional only"),
    "lfm2-causal": ("LFM2.5-VL-450M", "causal", "smaller + multimodal"),
    "lfm2-block": ("LFM2.5-VL-450M", "block", "smaller + bidirectional"),
}


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def num(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def load(path):
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="Directories containing arm subdirectories")
    ap.add_argument("--mm", help="mm_probe.py output")
    ap.add_argument("--djev", help="djev_zeroshot.py output")
    ap.add_argument("--mlx", help="mlx_latency.py output")
    ap.add_argument("--mlx-adapters", help="mlx_qev_adapters.py output")
    ap.add_argument("--order", help="order_probe.py output")
    ap.add_argument("--published", default="docs/results/snake_training", help="Published Qev Snake results")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    digest, lines = {"arms": {}}, []
    lines += [("| Arm | Backbone | Attention | Params | Dev acc | Snake acc | General acc | Dev NLL | ECE | "
               "Snake food (20 games) | Collisions | GPU ms/decision (adapter / merged) | Train min |"),
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    published = load(Path(args.published) / "evaluation.json")
    closed = load(Path(args.published) / "trained-snake-torch.json")
    if published:
        dev = published["development"]["calibrated"]
        snake = published["development"]["by_source"].get("synthetic_snake_observable_teacher", {})
        general = load(Path(args.published) / "training_metrics_summary.json")["trained_general"]
        lines.append(f"| published qev-snake-0.8b | Qwen3.5-0.8B (two-stage) | causal | 0.86B | {pct(dev['accuracy'])} | "
                     f"{pct(snake.get('accuracy'))} | {pct(general['accuracy'])} | {num(dev['nll'])} | {num(dev['ece'])} | "
                     f"{closed['aggregate']['food']['mean']:.1f} | {closed['aggregate']['collisions']} | — | — |")
    order = list(ARMS)
    for root in args.runs:
        for arm_dir in sorted(Path(root).iterdir(), key=lambda d: (order.index(d.name) if d.name in order else len(order), d.name)):
            evaluation = load(arm_dir / "evaluation.json")
            if not evaluation:
                continue
            label = evaluation["label"]
            latency, provenance = load(arm_dir / "latency.json") or {}, load(arm_dir / "provenance.json") or {}
            dev = evaluation["development"]
            loop = evaluation.get("snake_closed_loop", {})
            gpu = (latency.get("adapter", {}).get("single_snake_ms", {}).get("p50"),
                   latency.get("merged", {}).get("single_snake_ms", {}).get("p50"))
            backbone, attention, _ = ARMS.get(label, (evaluation["backbone"], evaluation["attention"], ""))
            params = provenance.get("total_parameters")
            lines.append(f"| {label} | {backbone} | {attention} | {params / 1e9:.2f}B | {pct(dev['calibrated']['accuracy'])} | "
                         f"{pct(dev['snake']['accuracy'])} | {pct(dev['general']['accuracy'])} | {num(dev['calibrated']['nll'])} | "
                         f"{num(dev['calibrated']['ece'])} | {num(loop.get('food', {}).get('mean'), 1)} | "
                         f"{loop.get('collisions', '—')} | {num(gpu[0], 0)} / {num(gpu[1], 0)} | "
                         f"{evaluation['training_seconds'] / 60:.0f} |")
            digest["arms"][label] = {"development": dev["calibrated"], "snake": dev["snake"], "general": dev["general"],
                                     "temperature": dev["temperature"], "snake_closed_loop": loop, "latency": latency,
                                     "total_parameters": params, "training_seconds": evaluation["training_seconds"],
                                     "by_source": {k: v.get("accuracy") for k, v in dev["by_source"].items()}}
    djev = load(args.djev)
    if djev:
        dev = djev["development"]
        lines.append(f"| djev zero-shot* | DiffusionGemma-26B-A4B NVFP4 | 1-step diffusion read | 25.2B | "
                     f"{pct(dev['calibrated']['accuracy'])} | {pct(dev['snake']['accuracy'])} | {pct(dev['general']['accuracy'])} | "
                     f"{num(dev['calibrated']['nll'])} | {num(dev['calibrated']['ece'])} | — | — | "
                     f"{num(djev['serial_request_ms']['p50'], 0)} (HTTP) | 0 |")
        lines.append("")
        lines.append("\\* djev skips the 60 Banking77 questions (77 options exceed its 62 single-token labels); "
                     "its general accuracy covers the other 820.")
        digest["djev"] = {k: v for k, v in djev.items() if k != "development_rows"}
    order = load(args.order)
    if order:
        lines += ["", "Candidate-order robustness (all 1,380 development questions, options reversed):", "",
                  "| Checkpoint | Choice changes | Accuracy (original / reversed) | Mean max prob. shift | Banking77 changes |",
                  "|---|---|---|---|---|"]
        for label, arm in order.items():
            lines.append(f"| {label} | {pct(arm['choice_flip_rate'])} | {pct(arm['accuracy_original'])} / "
                         f"{pct(arm['accuracy_reversed'])} | {num(arm['mean_max_probability_shift'])} | "
                         f"{pct(arm['flip_rate_by_source'].get('banking77'))} |")
        digest["order_probe"] = order
    mm = load(args.mm)
    if mm:
        lines += ["", f"Image decisions, A-OKVQA validation ({mm['rows']} questions, 4 options, chance 25%):", "",
                  "| Checkpoint | Decision head + image | Decision head, no image | Frozen foundation letter read | ms/decision (GPU) |",
                  "|---|---|---|---|---|"]
        for label, arm in mm["arms"].items():
            lines.append(f"| {label} | {pct(arm['decision_accuracy'])} | {pct(arm['decision_no_image_accuracy'])} | "
                         f"{pct(arm['letter_accuracy'])} | {num(arm['decision_ms_p50'], 0)} |")
        digest["mm_probe"] = mm
    mlx = load(args.mlx)
    if mlx:
        qev = mlx.get("qev_production", {})
        lines += ["", f"Apple Silicon MLX, one Snake decision ({mlx['requests']} development states, p50 ms):", "",
                  "| Model | fp32 | bf16 | 8-bit |", "|---|---|---|---|"]
        if qev:
            lines.append(f"| Qev Qwen3.5-0.8B, switchable LoRA (production) | {num(qev['predict_logits_ms']['p50'], 0)} | "
                         f"{num(qev.get('predict_logits_bf16_ms', {}).get('p50'), 0)} | {num(qev.get('predict_logits_q8_ms', {}).get('p50'), 0)} |")
        adapters = load(args.mlx_adapters)
        if adapters:
            lines.append(f"| Qev Qwen3.5-0.8B, adapters off (merged-LoRA floor) | {num(adapters['adapters_off_fp32_ms']['p50'], 0)} | "
                         f"{num(adapters['adapters_off_bf16_ms']['p50'], 0)} | — |")
        for name, label in (("qwen3", "Qwen3-0.6B, merged LoRA"), ("lfm2-vl", "LFM2.5-VL-450M, merged LoRA")):
            if name in mlx:
                lines.append(f"| {label} | " + " | ".join(num(mlx[name].get(p, {}).get("causal_ms", {}).get("p50"), 0)
                                                         for p in ("fp32", "bf16", "q8")) + " |")
        digest["mlx"] = mlx
    text = "\n".join(lines)
    print(text)
    Path(args.output).write_text(json.dumps(digest, indent=2) + "\n")


if __name__ == "__main__":
    main()
