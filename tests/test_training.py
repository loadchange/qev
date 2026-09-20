"""Training/evaluation contract checks without model downloads or large weights."""

import math
from types import SimpleNamespace

import pytest
import torch

from qev.evaluation import (
    encode_records,
    fit_temperature,
    inference_autocast,
    metrics,
    predict_items,
    prediction_precision,
    report,
)
from qev.train import accumulation_loss


def test_uneven_microbatches_match_the_full_group_gradient():
    """The last 1-item batch must not receive the weight of a full 3-item batch."""
    features = torch.tensor([[1., 0.], [2., 1.], [-1., 2.], [4., 1.], [1., 3.]])
    labels = torch.tensor([0, 1, 1, 0, 1])
    weight = torch.tensor([[.3, -.2], [.1, .4]], requires_grad=True)
    expected = torch.autograd.grad(
        torch.nn.functional.cross_entropy(features @ weight, labels), weight
    )[0]
    for start, stop in ((0, 3), (3, 4), (4, 5)):
        accumulation_loss(features[start:stop] @ weight, labels[start:stop], 5).backward()
    torch.testing.assert_close(weight.grad, expected, atol=1e-7, rtol=1e-6)
    # A final accumulation group with one question uses its actual size.
    weight.grad = None
    accumulation_loss(features[-1:] @ weight, labels[-1:], 1).backward()
    expected_last = torch.autograd.grad(
        torch.nn.functional.cross_entropy(features[-1:] @ weight, labels[-1:]), weight
    )[0]
    torch.testing.assert_close(weight.grad, expected_last)


def test_metrics_are_outcome_based_and_do_not_cap_confident_errors():
    rows = [
        {"logits": [1000., 0.], "label": 1, "qtype": "choice"},
        {"logits": [0., 0., 0.], "label": 1, "qtype": "score"},
    ]
    result = metrics(rows)
    assert result["nll"] == pytest.approx((1000 + math.log(3)) / 2)
    assert result["brier"] == pytest.approx((2 + 2 / 3) / 2)
    assert result["score_mae"] == pytest.approx(0)
    assert result["confident_error_rate"] == .5


def test_calibration_uses_only_supplied_calibration_outcomes():
    calibration = [{"logits": [4., 0.], "label": int(i < 2)} for i in range(10)]
    development = [{"logits": [4., 0.], "label": 1} for _ in range(10)]
    temperature = fit_temperature(calibration)
    assert metrics(calibration, temperature)["nll"] <= metrics(calibration)["nll"]
    # Reporting a held-out split evaluates the fixed temperature; it must not refit.
    summary = report(development, temperature)
    assert summary["temperature"] == temperature
    assert summary["calibrated"] == metrics(development, temperature)
    assert fit_temperature(development) != temperature


@pytest.mark.parametrize("row", [
    {"logits": [float("nan"), 0.], "label": 0},
    {"logits": [0., float("inf")], "label": 0},
    {"logits": [], "label": 0},
    {"logits": [0., 1.], "label": 3},
    {"logits": [0., 1.], "label": .5},
])
def test_bad_predictions_never_produce_plausible_metrics(row):
    with pytest.raises(ValueError):
        metrics([row])


def test_empty_calibration_and_invalid_temperature_are_rejected():
    with pytest.raises(ValueError, match="calibration"):
        fit_temperature([])
    for temperature in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            metrics([{"logits": [0., 1.], "label": 1}], temperature)


class EchoCandidateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, option_positions, option_mask, decision_positions):
        # A tiny deterministic model lets the test focus on reordering and
        # candidate slicing, rather than duplicating the actual neural network.
        values = input_ids[:, :1].float() + torch.arange(option_positions.shape[1])[None]
        return values.masked_fill(~option_mask, torch.finfo(torch.float32).min)


def test_prediction_bucketing_restores_ids_and_excludes_padded_candidates():
    def item(first, length, options, index):
        return {
            "ids": [first] * length, "option_positions": list(range(options)),
            "decision_position": length - 1, "label": 0,
            "source": "rules", "language": "zh", "qtype": "score",
            "record_id": f"record-{index}", "question_id": "decision", "state_truncated": False,
        }
    inputs = [item(3, 9, 3, 0), item(7, 5, 2, 1), item(2, 7, 1, 2)]
    rows = predict_items(EchoCandidateModel(), inputs, SimpleNamespace(pad_token_id=0), batch_size=2)
    assert [row["record_id"] for row in rows] == ["record-0", "record-1", "record-2"]
    assert [row["logits"] for row in rows] == [[3., 4., 5.], [7., 8.], [2.]]
    assert all(row["language"] == "zh" and row["qtype"] == "score" for row in rows)
    assert report(rows)["by_language"]["zh"]["n"] == 3


def test_cuda_precision_policy_uses_bf16_even_with_fp32_master_weights(monkeypatch):
    model = SimpleNamespace(parameters=lambda: iter([
        SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float32)
    ]))
    received = {}
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: received.update(kwargs))
    inference_autocast(model)
    assert received == {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}
    policy = prediction_precision(model)
    assert policy["autocast_dtype"] == "bfloat16"
    assert policy["weight_dtype"] == policy["pointer_dtype"] == "float32"


def test_encode_records_keeps_training_language_and_question_types(monkeypatch):
    from qev import tokenization

    monkeypatch.setattr(tokenization, "encode_question", lambda *args: {
        "ids": [1, 2], "option_positions": [0, 1], "decision_position": 1,
        "state_truncated": False,
    })
    requests = [{
        "state": "客户要求退款。",
        "questions": {
            "eligible": {"type": "noul", "instructions": "可以退款吗？", "label": True, "src": "refund"},
            "severity": {"type": "score", "instructions": "严重程度？", "criteria": ["低", "高"], "label": 1},
        },
        "_meta": {"id": "r1", "source": "rules", "language": "zh"},
    }]
    encoded = encode_records(requests, None)
    assert [(row["qtype"], row["question_id"], row["source"], row["language"], row["label"]) for row in encoded] == [
        ("noul", "eligible", "refund", "zh", 1),
        ("score", "severity", "rules", "zh", 1),
    ]
