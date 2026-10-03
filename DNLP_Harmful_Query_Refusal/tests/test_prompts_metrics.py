from __future__ import annotations

import pytest

from pathlib import Path

from dnlp_refusal.metrics import classification_metrics, generation_metrics
from dnlp_refusal.models.flan_t5 import _validation_metrics_from_predictions
from dnlp_refusal.prompts import generic_refusal, parse_generation, training_target
from dnlp_refusal.types import Example, Prediction


def _example(uid: str, label: str, reference: str | None = None) -> Example:
    return Example(
        uid=uid,
        text="Invented toy request.",
        label=label,
        reference=reference,
        source="toy.json",
        source_index=0,
    )


def test_parser_distinguishes_harmless_from_harmful() -> None:
    harmless = parse_generation("LABEL: harmless\nREASON: ALLOW")
    harmful = parse_generation("LABEL: harmful\nREASON: Request refused.")
    assert harmless.label == "harmless"
    assert harmless.rationale == "ALLOW"
    assert not harmless.parse_error
    assert harmful.label == "harmful"
    assert not harmful.parse_error


def test_parser_accepts_one_line_label_and_reason() -> None:
    parsed = parse_generation(
        "LABEL: harmful REASON: Request refused for a safe toy reason."
    )
    assert parsed.label == "harmful"
    assert parsed.rationale == "Request refused for a safe toy reason."
    assert not parsed.parse_error


def test_parser_fails_closed() -> None:
    parsed = parse_generation("Unstructured model output")
    assert parsed.label == "harmful"
    assert parsed.parse_error
    assert parsed.rationale


def test_training_target_has_joint_label_and_reason() -> None:
    harmful = _example("a", "harmful", "Toy refusal reason.")
    harmless = _example("b", "harmless")
    assert training_target(harmful) == (
        "LABEL: harmful\nREASON: Toy refusal reason."
    )
    assert training_target(harmless) == "LABEL: harmless\nREASON: ALLOW"


def test_classification_metrics_capture_safety_errors() -> None:
    examples = [
        _example("a", "harmful", "Refuse."),
        _example("b", "harmful", "Refuse."),
        _example("c", "harmless"),
        _example("d", "harmless"),
    ]
    predictions = [
        Prediction("a", "harmful", "Refuse."),
        Prediction("b", "harmless", "ALLOW"),
        Prediction("c", "harmful", generic_refusal("delete")),
        Prediction("d", "harmless", "ALLOW"),
    ]
    metrics = classification_metrics(examples, predictions)
    assert metrics["harmful_precision"] == pytest.approx(0.5)
    assert metrics["harmful_recall"] == pytest.approx(0.5)
    assert metrics["harmful_f1"] == pytest.approx(0.5)
    assert metrics["unsafe_compliance_rate"] == pytest.approx(0.5)
    assert metrics["over_refusal_rate"] == pytest.approx(0.5)


def test_prediction_order_is_checked() -> None:
    examples = [_example("a", "harmful", "Refuse.")]
    predictions = [Prediction("wrong", "harmful", "Refuse.")]
    with pytest.raises(ValueError, match="order or IDs"):
        classification_metrics(examples, predictions)


def test_skipped_bertscore_does_not_require_references() -> None:
    examples = [
        _example("a", "harmful"),
        _example("b", "harmless"),
    ]
    predictions = [
        Prediction("a", "harmful", "Generic local refusal."),
        Prediction("b", "harmless", "ALLOW"),
    ]

    metrics = generation_metrics(
        examples,
        predictions,
        bertscore_model_dir=Path("unused-when-skipped"),
        bertscore_num_layers=17,
        device="cpu",
        skip_bertscore=True,
    )

    assert metrics["bertscore_f1_end_to_end"] is None
    assert metrics["n_harmful_references"] == 0
    assert metrics["n_correctly_refused"] == 1


def test_flan_checkpoint_selection_rejects_parse_error_collapse() -> None:
    examples = [
        _example("a", "harmful"),
        _example("b", "harmful"),
        _example("c", "harmless"),
        _example("d", "harmless"),
    ]
    collapsed = [
        Prediction(item.uid, "harmful", "Generic refusal.", parse_error=True)
        for item in examples
    ]
    balanced = [
        Prediction(item.uid, item.label, "Valid output.", parse_error=False)
        for item in examples
    ]

    collapsed_metrics = _validation_metrics_from_predictions(examples, collapsed)
    balanced_metrics = _validation_metrics_from_predictions(examples, balanced)

    assert collapsed_metrics["validation_harmful_f1"] == pytest.approx(2 / 3)
    assert collapsed_metrics["validation_selection_score"] < 0
    assert balanced_metrics["validation_selection_score"] == pytest.approx(1.0)
