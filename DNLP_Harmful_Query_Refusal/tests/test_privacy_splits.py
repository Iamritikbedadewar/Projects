from __future__ import annotations

import socket
from pathlib import Path

import pytest

from dnlp_refusal.privacy import (
    NetworkBlocker,
    PrivacyError,
    validate_local_data_dir,
)
from dnlp_refusal.splits import make_folds
from dnlp_refusal.types import Example


def test_network_blocker_denies_socket_before_connection() -> None:
    with NetworkBlocker(enabled=True):
        with pytest.raises(PrivacyError, match="Outbound network access"):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)


def test_cloud_sync_marker_is_rejected(tmp_path: Path) -> None:
    cloud_path = tmp_path / "OneDrive" / "private"
    cloud_path.mkdir(parents=True)
    with pytest.raises(PrivacyError, match="cloud-synchronized"):
        validate_local_data_dir(cloud_path)


def test_stratified_folds_have_no_id_overlap() -> None:
    examples = [
        Example(
            uid=f"h-{index}",
            text=f"Harmful toy {index}",
            label="harmful",
            reference="Refused.",
            source="toy",
            source_index=index,
        )
        for index in range(10)
    ]
    examples.extend(
        Example(
            uid=f"s-{index}",
            text=f"Harmless toy {index}",
            label="harmless",
            reference=None,
            source="toy",
            source_index=index,
        )
        for index in range(10)
    )
    folds = make_folds(
        examples,
        n_splits=5,
        validation_fraction=0.25,
        seed=42,
    )
    assert len(folds) == 5
    test_ids = [example.uid for fold in folds for example in fold.test]
    assert len(test_ids) == len(set(test_ids)) == len(examples)
    for fold in folds:
        assert {example.label for example in fold.test} == {"harmful", "harmless"}

