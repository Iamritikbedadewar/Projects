from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .types import Example


@dataclass(frozen=True)
class Fold:
    number: int
    train: list[Example]
    validation: list[Example]
    test: list[Example]


def make_folds(
    examples: Sequence[Example],
    *,
    n_splits: int,
    validation_fraction: float,
    seed: int,
) -> list[Fold]:
    labels = np.array([example.label for example in examples])
    indices = np.arange(len(examples))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[Fold] = []

    for fold_number, (development_idx, test_idx) in enumerate(
        splitter.split(indices, labels)
    ):
        development_labels = labels[development_idx]
        train_idx, validation_idx = train_test_split(
            development_idx,
            test_size=validation_fraction,
            random_state=seed + fold_number,
            stratify=development_labels,
        )
        fold = Fold(
            number=fold_number,
            train=[examples[int(index)] for index in train_idx],
            validation=[examples[int(index)] for index in validation_idx],
            test=[examples[int(index)] for index in test_idx],
        )
        _assert_no_leakage(fold)
        folds.append(fold)
    return folds


def _assert_no_leakage(fold: Fold) -> None:
    groups = [
        {example.uid for example in fold.train},
        {example.uid for example in fold.validation},
        {example.uid for example in fold.test},
    ]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError(f"Data leakage detected in fold {fold.number}.")

