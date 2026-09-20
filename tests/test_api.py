import math

import pytest
from pydantic import ValidationError

from qev.api import SystemOneRequest, render, to_answers, to_record


def test_typed_request_renders_structured_content_and_maps_raw_probabilities():
    request = {"state": {"customer": "Mira", "events": ["double charge", "refund"]}, "questions": {
        "department": {"type": "choice", "instructions": "Choose a team", "criteria": {"billing": "Charges", "sales": None}},
        "refund": {"type": "noul", "instructions": {"rule": "Money requested back"}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "Medium", "High"]},
    }}
    record, metadata = to_record(request)
    assert record["state"] == "customer: Mira\nevents:\n  - double charge\n  - refund"
    assert record["questions"][0]["options"] == ["billing: Charges", "sales"]
    assert record["questions"][1]["keys"] == ["false", "true"]
    assert record["questions"][1]["instr"] == "rule: Money requested back"
    answers = to_answers([[0.87654321, 0.12345679], [0.2, 0.8], [0.1, 0.2, 0.7]], metadata)
    assert answers["department"]["choice"] == "billing"
    assert answers["department"]["probabilities"]["billing"] == 0.87654321
    assert answers["refund"] == {"type": "noul", "noul": 0.8}
    assert answers["urgency"]["score"] == pytest.approx(1.6)
    assert answers["urgency"]["legend"] == {"0": "Low", "1": "Medium", "2": "High"}


@pytest.mark.parametrize("count", [1, 255])
def test_choice_boundary_counts(count):
    record, metadata = to_record({"state": "", "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {str(i): None for i in range(count)}}}})
    answer = to_answers([[1 / count] * count], metadata)["q"]
    assert len(record["questions"][0]["options"]) == count
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    assert answer["confidence"] == pytest.approx(1 if count == 1 else 0)


@pytest.mark.parametrize("kind,criteria", [("choice", {}), ("choice", {str(i): None for i in range(256)}), ("score", ["one"]), ("score", ["x"] * 256), ("noul", {"maybe": "unknown"})])
def test_invalid_option_schema_is_rejected(kind, criteria):
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "", "questions": {"q": {"type": kind, "instructions": "", "criteria": criteria}}})


@pytest.mark.parametrize("values", [[math.nan, 1], [-0.2, 1.2], [0.2, 0.3], [1], [0, 0]])
def test_invalid_model_probabilities_fail_closed(values):
    _, metadata = to_record({"state": "", "questions": {"q": {"type": "noul", "instructions": "True?"}}})
    with pytest.raises(ValueError):
        to_answers([values], metadata)


def test_probability_count_mismatch_is_not_silently_zipped():
    _, metadata = to_record({"state": "", "questions": {"q": {"type": "noul", "instructions": "True?"}}})
    with pytest.raises(ValueError):
        to_answers([], metadata)


def test_json_renderer_handles_booleans_without_python_spellings():
    assert render({"flag": True, "missing": None}) == "flag: true\nmissing: "
    with pytest.raises(ValueError):
        render(float("inf"))
