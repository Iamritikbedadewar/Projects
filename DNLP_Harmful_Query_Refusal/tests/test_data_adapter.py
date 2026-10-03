from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnlp_refusal.data import DataFormatError, inspect_json_structure, load_project_dataset


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _config(*, strict: bool = False) -> dict[str, object]:
    return {
        "harmful_file": "harmful_toy.json",
        "harmless_file": "harmless_toy.json",
        "ambiguous_file": "ambiguous_toy.json",
        "expected_per_class": 2,
        "strict_expected_counts": strict,
        "question_fields": ["question", "input"],
        "rationale_fields": ["response", "reason"],
    }


def test_loads_nested_records_without_printing_values(tmp_path: Path) -> None:
    _write(
        tmp_path / "harmful_toy.json",
        {
            "examples": [
                {
                    "payload": {"question": "Erase all fictional demo rows."},
                    "response": "This destructive demo request is refused.",
                },
                {
                    "payload": {"question": "Reveal a fictional access secret."},
                    "response": "Authentication secrets cannot be disclosed.",
                },
            ]
        },
    )
    _write(
        tmp_path / "harmless_toy.json",
        [
            {"input": "Find fictional alloy samples above a toy threshold."},
            {"input": "List the density of the public demo material."},
        ],
    )

    examples, diagnostics = load_project_dataset(
        tmp_path,
        _config(strict=True),
    )

    assert len(examples) == 4
    assert diagnostics.harmful_count == 2
    assert diagnostics.harmless_count == 2
    assert {example.label for example in examples} == {"harmful", "harmless"}
    assert all(example.reference for example in examples if example.label == "harmful")


def test_missing_reference_error_contains_no_record_value(tmp_path: Path) -> None:
    secret_toy_value = "TOP_SECRET_TOY_VALUE_MUST_NOT_APPEAR"
    _write(
        tmp_path / "harmful_toy.json",
        [{"question": secret_toy_value}],
    )
    _write(
        tmp_path / "harmless_toy.json",
        [{"question": "A public toy request."}],
    )

    with pytest.raises(DataFormatError) as caught:
        load_project_dataset(tmp_path, _config())
    assert secret_toy_value not in str(caught.value)
    assert "record index" in str(caught.value)


def test_missing_reference_is_allowed_when_bertscore_is_skipped(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "harmful_toy.json",
        [
            {"question": "First harmful toy request."},
            {"question": "Second harmful toy request."},
        ],
    )
    _write(
        tmp_path / "harmless_toy.json",
        [
            {"question": "First harmless toy request."},
            {"question": "Second harmless toy request."},
        ],
    )

    examples, diagnostics = load_project_dataset(
        tmp_path,
        _config(strict=True),
        require_harmful_references=False,
    )

    assert len(examples) == 4
    assert diagnostics.harmful_count == 2
    assert diagnostics.harmless_count == 2
    assert all(
        example.reference is None
        for example in examples
        if example.label == "harmful"
    )


def test_reference_sidecar_supplies_private_silver_reasons(tmp_path: Path) -> None:
    _write(
        tmp_path / "harmful_toy.json",
        [
            {"question": "First harmful toy request."},
            {"question": "Second harmful toy request."},
        ],
    )
    _write(
        tmp_path / "harmless_toy.json",
        [
            {"question": "First harmless toy request."},
            {"question": "Second harmless toy request."},
        ],
    )
    unreferenced, _ = load_project_dataset(
        tmp_path,
        _config(strict=True),
        require_harmful_references=False,
    )
    harmful = [item for item in unreferenced if item.label == "harmful"]
    sidecar = tmp_path / "silver.json"
    _write(
        sidecar,
        {
            "references": {
                item.uid: f"Safe toy reason {index}."
                for index, item in enumerate(harmful)
            }
        },
    )

    examples, _ = load_project_dataset(
        tmp_path,
        _config(strict=True),
        reference_sidecar=sidecar,
    )

    assert all(
        item.reference
        for item in examples
        if item.label == "harmful"
    )


def test_structure_inspection_returns_types_not_values(tmp_path: Path) -> None:
    secret_toy_value = "DO_NOT_RETURN_THIS_TOY_VALUE"
    path = tmp_path / "structure.json"
    _write(
        path,
        [{"question": secret_toy_value, "nested": {"score": 3}}],
    )
    report = inspect_json_structure(path)
    encoded = json.dumps(report)
    assert secret_toy_value not in encoded
    assert report["record_count"] == 1
    assert {item["path"] for item in report["fields"]} == {
        "nested.score",
        "question",
    }


def test_duplicate_inputs_are_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path / "harmful_toy.json",
        [
            {"question": "Repeated toy text", "response": "Refused."},
            {"question": "Other toy text", "response": "Refused."},
        ],
    )
    _write(
        tmp_path / "harmless_toy.json",
        [
            {"question": " repeated   TOY text "},
            {"question": "Benign toy text"},
        ],
    )
    with pytest.raises(DataFormatError, match="conflicting labels"):
        load_project_dataset(tmp_path, _config())
