from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .types import Example


class DataFormatError(ValueError):
    """A structure-only error that must never contain a record value."""


@dataclass(frozen=True)
class DataDiagnostics:
    harmful_count: int
    harmless_count: int
    question_field_usage: dict[str, int]
    rationale_field_usage: dict[str, int]


CONTAINER_ALIASES = (
    "data",
    "items",
    "examples",
    "records",
    "pairs",
    "questions",
    "queries",
)


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _normalise_text_for_hash(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _stringify_candidate(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _looks_like_structured_query(value: str) -> bool:
    stripped = value.lstrip()
    lowered = stripped.casefold()
    return (
        stripped.startswith(("{", "["))
        or ".find(" in lowered
        or ".aggregate(" in lowered
        or '"$match"' in lowered
        or '"$project"' in lowered
    )


def _walk(record: Any, prefix: str = "") -> Iterator[tuple[str, str, Any]]:
    if isinstance(record, dict):
        for key, value in record.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            yield path, key_text, value
            yield from _walk(value, path)
    elif isinstance(record, list):
        for index, value in enumerate(record):
            path = f"{prefix}[{index}]"
            yield from _walk(value, path)


def _records_from_payload(payload: Any, filename: str) -> list[Any]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        direct_by_normalised = {
            _normalise_key(str(key)): value for key, value in payload.items()
        }
        for alias in CONTAINER_ALIASES:
            candidate = direct_by_normalised.get(_normalise_key(alias))
            if isinstance(candidate, list):
                return candidate
            if isinstance(candidate, dict) and candidate and all(
                isinstance(value, dict) for value in candidate.values()
            ):
                return list(candidate.values())

        values = list(payload.values())
        list_values = [value for value in values if isinstance(value, list)]
        if len(list_values) == 1:
            return list_values[0]
        if values and all(isinstance(value, dict) for value in values):
            return values

        if payload and all(isinstance(key, str) for key in payload) and all(
            isinstance(value, str) for value in values
        ):
            return [
                {"question": key, "response": value}
                for key, value in payload.items()
            ]

        return [payload]

    raise DataFormatError(
        f"{filename}: expected a JSON list or object at the document root."
    )


def _find_alias_value(
    record: Any,
    aliases: Sequence[str],
) -> tuple[str | None, str | None]:
    alias_order = {
        _normalise_key(alias): position for position, alias in enumerate(aliases)
    }
    matches: list[tuple[int, int, str, str]] = []

    if isinstance(record, str):
        return record.strip() or None, "<record>"

    for path, leaf_key, value in _walk(record):
        normalised = _normalise_key(leaf_key)
        if normalised not in alias_order:
            continue
        candidate = _stringify_candidate(value)
        if candidate is None:
            continue
        depth = path.count(".") + path.count("[")
        matches.append((alias_order[normalised], depth, path, candidate))

    if not matches:
        return None, None
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, path, value = matches[0]
    return value, path


def _available_leaf_fields(record: Any) -> list[str]:
    fields = {
        path
        for path, _leaf, value in _walk(record)
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return sorted(fields)


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except UnicodeDecodeError as exc:
        raise DataFormatError(f"{path.name}: the file is not valid UTF-8.") from exc
    except json.JSONDecodeError as exc:
        raise DataFormatError(
            f"{path.name}: invalid JSON near line {exc.lineno}, column {exc.colno}."
        ) from exc


def _make_uid(source: str, index: int, text: str) -> str:
    payload = f"{source}:{index}:{_normalise_text_for_hash(text)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_labelled_file(
    path: Path,
    *,
    label: str,
    question_fields: Sequence[str],
    rationale_fields: Sequence[str],
    require_reference: bool,
) -> tuple[list[Example], Counter[str], Counter[str]]:
    if not path.is_file():
        raise DataFormatError(f"Required file is missing: {path.name}")

    records = _records_from_payload(_read_json(path), path.name)
    examples: list[Example] = []
    question_usage: Counter[str] = Counter()
    rationale_usage: Counter[str] = Counter()
    missing_questions: list[int] = []
    missing_references: list[int] = []
    missing_question_fields: dict[int, list[str]] = {}

    for index, record in enumerate(records):
        text, question_path = _find_alias_value(record, question_fields)
        if (
            text is not None
            and question_path is not None
            and _normalise_key(question_path.rsplit(".", 1)[-1]) == "query"
            and _looks_like_structured_query(text)
        ):
            text = None
        if text is None:
            missing_questions.append(index)
            missing_question_fields[index] = _available_leaf_fields(record)
            continue

        reference, rationale_path = _find_alias_value(record, rationale_fields)
        if require_reference and reference is None:
            missing_references.append(index)

        question_usage[question_path or "<unknown>"] += 1
        if rationale_path:
            rationale_usage[rationale_path] += 1
        examples.append(
            Example(
                uid=_make_uid(path.name, index, text),
                text=text,
                label=label,
                reference=reference,
                source=path.name,
                source_index=index,
            )
        )

    if missing_questions:
        preview = ", ".join(str(index) for index in missing_questions[:10])
        field_summary = sorted(
            {
                field
                for index in missing_questions[:10]
                for field in missing_question_fields[index]
            }
        )
        raise DataFormatError(
            f"{path.name}: no configured question field in record index/indices "
            f"{preview}. Available leaf fields include: {', '.join(field_summary) or '<none>'}. "
            "No record values were printed."
        )

    if missing_references:
        preview = ", ".join(str(index) for index in missing_references[:10])
        raise DataFormatError(
            f"{path.name}: harmful record index/indices {preview} have no configured "
            "reference-rationale field. BERTScore requires a gold refusal reason. "
            "No record values were printed."
        )

    return examples, question_usage, rationale_usage


def _check_duplicates(examples: Iterable[Example]) -> None:
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[tuple[str, str]] = []
    conflicts: list[tuple[str, str]] = []

    for example in examples:
        digest = hashlib.sha256(
            _normalise_text_for_hash(example.text).encode("utf-8")
        ).hexdigest()
        if digest not in seen:
            seen[digest] = (example.label, example.uid)
            continue
        prior_label, prior_uid = seen[digest]
        if prior_label == example.label:
            duplicates.append((prior_uid, example.uid))
        else:
            conflicts.append((prior_uid, example.uid))

    if conflicts:
        pairs = ", ".join(f"{a}/{b}" for a, b in conflicts[:10])
        raise DataFormatError(
            "The same normalized input appears with conflicting labels. "
            f"Opaque ID pairs: {pairs}."
        )
    if duplicates:
        pairs = ", ".join(f"{a}/{b}" for a, b in duplicates[:10])
        raise DataFormatError(
            "Duplicate normalized inputs could leak across folds. "
            f"Opaque ID pairs: {pairs}."
        )


def _attach_reference_sidecar(
    examples: Sequence[Example],
    sidecar_path: Path,
) -> list[Example]:
    if not sidecar_path.is_file():
        raise DataFormatError(
            f"Reference sidecar is missing: {sidecar_path.name}"
        )
    payload = _read_json(sidecar_path)
    raw_references = (
        payload.get("references") if isinstance(payload, dict) else None
    )
    if not isinstance(raw_references, dict):
        raise DataFormatError(
            f"{sidecar_path.name}: expected a 'references' object."
        )
    if not all(
        isinstance(uid, str)
        and isinstance(reference, str)
        and bool(reference.strip())
        for uid, reference in raw_references.items()
    ):
        raise DataFormatError(
            f"{sidecar_path.name}: every reference must map an opaque string ID "
            "to a non-empty string."
        )

    expected_uids = {example.uid for example in examples}
    supplied_uids = set(raw_references)
    missing_count = len(expected_uids - supplied_uids)
    extra_count = len(supplied_uids - expected_uids)
    if missing_count or extra_count:
        raise DataFormatError(
            f"{sidecar_path.name}: reference-ID mismatch "
            f"(missing={missing_count}, unexpected={extra_count}). "
            "No record values were printed."
        )

    return [
        replace(example, reference=raw_references[example.uid].strip())
        for example in examples
    ]


def load_project_dataset(
    data_dir: Path,
    config: dict[str, Any],
    *,
    require_harmful_references: bool = True,
    reference_sidecar: Path | None = None,
) -> tuple[list[Example], DataDiagnostics]:
    """Load only the two files required for the core experiment."""

    question_fields = list(config["question_fields"])
    rationale_fields = list(config["rationale_fields"])
    harmful_path = data_dir / str(config["harmful_file"])
    harmless_path = data_dir / str(config["harmless_file"])

    harmful, harmful_q, harmful_r = _load_labelled_file(
        harmful_path,
        label="harmful",
        question_fields=question_fields,
        rationale_fields=rationale_fields,
        require_reference=False,
    )
    harmless, harmless_q, harmless_r = _load_labelled_file(
        harmless_path,
        label="harmless",
        question_fields=question_fields,
        rationale_fields=rationale_fields,
        require_reference=False,
    )

    if reference_sidecar is not None:
        harmful = _attach_reference_sidecar(harmful, reference_sidecar)

    if require_harmful_references:
        missing_references = [
            example.source_index
            for example in harmful
            if example.reference is None
        ]
        if missing_references:
            preview = ", ".join(str(index) for index in missing_references[:10])
            raise DataFormatError(
                f"{harmful_path.name}: harmful record index/indices {preview} "
                "have no configured reference-rationale field or sidecar entry. "
                "BERTScore requires a reference reason. "
                "No record values were printed."
            )

    expected = int(config.get("expected_per_class", 100))
    if bool(config.get("strict_expected_counts", True)):
        actual = {"harmful": len(harmful), "harmless": len(harmless)}
        wrong = {label: count for label, count in actual.items() if count != expected}
        if wrong:
            details = ", ".join(f"{label}={count}" for label, count in wrong.items())
            raise DataFormatError(
                f"Expected {expected} examples per class but found {details}. "
                "Check the protocol and adapter fields before training."
            )

    examples = harmful + harmless
    _check_duplicates(examples)

    question_usage = harmful_q + harmless_q
    rationale_usage = harmful_r + harmless_r
    diagnostics = DataDiagnostics(
        harmful_count=len(harmful),
        harmless_count=len(harmless),
        question_field_usage=dict(sorted(question_usage.items())),
        rationale_field_usage=dict(sorted(rationale_usage.items())),
    )
    return examples, diagnostics


def inspect_json_structure(path: Path) -> dict[str, Any]:
    """Return counts, field paths, and value types without returning values."""

    payload = _read_json(path)
    records = _records_from_payload(payload, path.name)
    fields: Counter[str] = Counter()
    types: dict[str, set[str]] = {}
    for record in records:
        for field_path, _leaf, value in _walk(record):
            if isinstance(value, (dict, list)):
                continue
            fields[field_path] += 1
            types.setdefault(field_path, set()).add(type(value).__name__)
    return {
        "file": path.name,
        "record_count": len(records),
        "fields": [
            {
                "path": field,
                "present_in_records": count,
                "types": sorted(types.get(field, set())),
            }
            for field, count in sorted(fields.items())
        ],
    }
