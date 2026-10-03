from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import platform
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import load_config, project_root, resolve_project_path
from .data import DataDiagnostics, load_project_dataset
from .metrics import evaluate_predictions
from .privacy import NetworkBlocker, enable_offline_environment, validate_local_data_dir

# This also protects users who invoke the module directly rather than using the
# PowerShell wrapper. It runs before Transformers is imported below.
enable_offline_environment()

from .models.flan_t5 import fine_tune_local_flan, load_local_flan
from .models.kim_cnn import train_kimcnn
from .splits import Fold, make_folds
from .types import Example, Prediction


SUPPORTED_MODELS = ("kimcnn", "flan_zero", "flan_finetuned")
HEADLINE_METRICS = (
    "harmful_f1",
    "macro_f1",
    "harmful_precision",
    "harmful_recall",
    "over_refusal_rate",
    "unsafe_compliance_rate",
    "bertscore_f1_end_to_end",
    "bertscore_f1_when_refused",
    "parse_error_rate",
)


def _parse_csv_argument(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_models(value: str) -> list[str]:
    models = _parse_csv_argument(value)
    unknown = sorted(set(models) - set(SUPPORTED_MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(
            "Unknown model(s): "
            + ", ".join(unknown)
            + ". Choose from "
            + ", ".join(SUPPORTED_MODELS)
        )
    if not models:
        raise argparse.ArgumentTypeError("At least one model is required.")
    return models


def _parse_folds(value: str) -> list[int]:
    try:
        folds = [int(item) for item in _parse_csv_argument(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Folds must be comma-separated integers.") from exc
    if not folds:
        raise argparse.ArgumentTypeError("At least one fold is required.")
    return folds


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(value)


def _metrics_for(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    *,
    config: dict[str, Any],
    device: torch.device,
    skip_bertscore: bool,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    return evaluate_predictions(
        examples,
        predictions,
        bertscore_model_dir=resolve_project_path(
            evaluation["bertscore_model_dir"]
        ),
        bertscore_num_layers=int(evaluation["bertscore_num_layers"]),
        device=str(device),
        skip_bertscore=skip_bertscore,
    )


def _save_private_errors(
    path: Path,
    *,
    fold: Fold,
    model_name: str,
    predictions: Sequence[Prediction],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "model",
                "uid",
                "gold_label",
                "predicted_label",
                "parse_error",
                "input_text",
                "reference_rationale",
                "generated_rationale",
            ],
        )
        if handle.tell() == 0:
            writer.writeheader()
        for example, prediction in zip(fold.test, predictions, strict=True):
            if example.label == prediction.label and not prediction.parse_error:
                continue
            writer.writerow(
                {
                    "fold": fold.number,
                    "model": model_name,
                    "uid": example.uid,
                    "gold_label": example.label,
                    "predicted_label": prediction.label,
                    "parse_error": prediction.parse_error,
                    "input_text": example.text,
                    "reference_rationale": example.reference or "",
                    "generated_rationale": prediction.rationale,
                }
            )


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1) if len(array) > 1 else 0.0)


def _aggregate(
    fold_results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for model_name, results in fold_results.items():
        summary: dict[str, Any] = {"n_folds": len(results)}
        for metric in HEADLINE_METRICS:
            values = [
                float(result[metric])
                for result in results
                if result.get(metric) is not None
            ]
            if not values:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
                continue
            mean, std = _mean_std(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std

        matrices = [
            np.asarray(result["confusion_matrix"], dtype=int) for result in results
        ]
        summary["confusion_matrix_labels"] = ["harmless", "harmful"]
        summary["confusion_matrix_sum"] = np.sum(matrices, axis=0).tolist()
        aggregated[model_name] = summary
    return aggregated


def _write_summary_csv(
    path: Path,
    aggregated: dict[str, dict[str, Any]],
) -> None:
    fieldnames = ["model"]
    for metric in HEADLINE_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, summary in aggregated.items():
            row = {"model": model_name}
            row.update({field: summary.get(field) for field in fieldnames[1:]})
            writer.writerow(row)


def _plot_headline_metrics(
    path: Path,
    aggregated: dict[str, dict[str, Any]],
) -> None:
    metrics = [
        ("harmful_f1", "Harmful F1"),
        ("macro_f1", "Macro F1"),
        ("bertscore_f1_end_to_end", "BERTScore F1"),
    ]
    model_names = list(aggregated)
    x = np.arange(len(model_names))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5.2))
    colors = ("#22577A", "#38A3A5", "#F4A261")

    plotted = 0
    for index, (metric, label) in enumerate(metrics):
        means = [aggregated[name].get(f"{metric}_mean") for name in model_names]
        if all(value is None for value in means):
            continue
        heights = [0.0 if value is None else float(value) for value in means]
        errors = [
            0.0
            if aggregated[name].get(f"{metric}_std") is None
            else float(aggregated[name][f"{metric}_std"])
            for name in model_names
        ]
        offset = (plotted - 1) * width
        axis.bar(
            x + offset,
            heights,
            width,
            yerr=errors,
            capsize=3,
            label=label,
            color=colors[index],
        )
        plotted += 1

    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(x)
    axis.set_xticklabels(model_names)
    axis.set_title("Model comparison across selected folds")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_results(
    public_dir: Path,
    *,
    run_id: str,
    diagnostics: DataDiagnostics,
    selected_folds: Sequence[int],
    fold_results: dict[str, list[dict[str, Any]]],
    device: torch.device,
    settings: dict[str, Any],
) -> None:
    aggregated = _aggregate(fold_results)
    version_packages = (
        "torch",
        "transformers",
        "scikit-learn",
        "numpy",
        "bert-score",
    )
    package_versions: dict[str, str] = {}
    for package in version_packages:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"

    manifest_path = project_root() / "models" / "MODEL_MANIFEST.json"
    public_model_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            public_model_manifest = json.load(handle)

    payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions,
        },
        "public_model_manifest": public_model_manifest,
        "settings": settings,
        "dataset_counts": {
            "harmful": diagnostics.harmful_count,
            "harmless": diagnostics.harmless_count,
        },
        "folds": list(selected_folds),
        "per_fold": fold_results,
        "aggregate": aggregated,
    }
    with (public_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    _write_summary_csv(public_dir / "summary.csv", aggregated)
    _plot_headline_metrics(public_dir / "model_comparison.png", aggregated)


def _precompute_zero_shot(
    examples: Sequence[Example],
    *,
    model_dir: Path,
    model_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Prediction]:
    print("Running the zero-shot FLAN-T5 baseline locally.")
    bundle = load_local_flan(model_dir, config=model_config, device=device)
    predictions = bundle.generate(examples)
    result = {prediction.uid: prediction for prediction in predictions}
    del bundle
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> Path:
    enable_offline_environment()
    config = load_config(args.config)
    data_dir = validate_local_data_dir(
        args.data_dir,
        reject_cloud_synced_paths=bool(
            config["privacy"].get("reject_cloud_synced_paths", True)
        ),
    )
    device = _device_from_argument(args.device)

    root = project_root()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    public_dir = resolve_project_path(config["outputs"]["public_dir"]) / run_id
    private_dir = resolve_project_path(config["outputs"]["private_dir"]) / run_id
    public_dir.mkdir(parents=True, exist_ok=False)
    private_dir.mkdir(parents=True, exist_ok=False)

    block_network = bool(config["privacy"].get("block_network", True))
    with NetworkBlocker(enabled=block_network):
        examples, diagnostics = load_project_dataset(
            data_dir,
            config["data"],
            require_harmful_references=not args.skip_bertscore,
            reference_sidecar=(
                Path(args.reference_sidecar).expanduser().resolve()
                if args.reference_sidecar
                else None
            ),
        )
        print(
            "Loaded the expected confidential dataset in memory: "
            f"{diagnostics.harmful_count} harmful and "
            f"{diagnostics.harmless_count} harmless examples."
        )

        folds = make_folds(
            examples,
            n_splits=int(config["project"]["n_splits"]),
            validation_fraction=float(config["project"]["validation_fraction"]),
            seed=int(config["project"]["seed"]),
        )
        invalid_folds = sorted(set(args.folds) - {fold.number for fold in folds})
        if invalid_folds:
            raise ValueError(f"Invalid fold number(s): {invalid_folds}")
        selected = [fold for fold in folds if fold.number in args.folds]

        flan_config = dict(config["models"]["flan_t5"])
        flan_config["prompt_variant"] = args.prompt_variant
        flan_config["rationale_mode"] = args.rationale_mode
        flan_model_dir = root / str(flan_config["model_dir"])
        zero_by_uid: dict[str, Prediction] = {}
        if "flan_zero" in args.models:
            zero_by_uid = _precompute_zero_shot(
                examples,
                model_dir=flan_model_dir,
                model_config=flan_config,
                device=device,
            )

        fold_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        training_history: dict[str, dict[str, Any]] = defaultdict(dict)
        private_error_path = private_dir / "private_error_analysis.csv"

        for fold in selected:
            print(f"Processing fold {fold.number}.")
            if "kimcnn" in args.models:
                bundle = train_kimcnn(
                    fold.train,
                    fold.validation,
                    config=config["models"]["kimcnn"],
                    seed=int(config["project"]["seed"]) + fold.number,
                    device=device,
                    checkpoint_path=private_dir
                    / "checkpoints"
                    / f"kimcnn_fold_{fold.number}.pt",
                )
                predictions = bundle.predict(fold.test)
                metrics = _metrics_for(
                    fold.test,
                    predictions,
                    config=config,
                    device=device,
                    skip_bertscore=args.skip_bertscore,
                )
                metrics["fold"] = fold.number
                fold_results["kimcnn"].append(metrics)
                training_history["kimcnn"][str(fold.number)] = bundle.training_history
                if args.save_private_errors:
                    _save_private_errors(
                        private_error_path,
                        fold=fold,
                        model_name="kimcnn",
                        predictions=predictions,
                    )
                del bundle

            if "flan_zero" in args.models:
                predictions = [zero_by_uid[example.uid] for example in fold.test]
                metrics = _metrics_for(
                    fold.test,
                    predictions,
                    config=config,
                    device=device,
                    skip_bertscore=args.skip_bertscore,
                )
                metrics["fold"] = fold.number
                fold_results["flan_zero"].append(metrics)
                if args.save_private_errors:
                    _save_private_errors(
                        private_error_path,
                        fold=fold,
                        model_name="flan_zero",
                        predictions=predictions,
                    )

            if "flan_finetuned" in args.models:
                checkpoint_dir = (
                    private_dir
                    / "checkpoints"
                    / f"flan_t5_finetuned_fold_{fold.number}"
                )
                bundle, history = fine_tune_local_flan(
                    flan_model_dir,
                    fold.train,
                    fold.validation,
                    config=flan_config,
                    seed=int(config["project"]["seed"]) + fold.number,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                )
                predictions = bundle.generate(fold.test)
                metrics = _metrics_for(
                    fold.test,
                    predictions,
                    config=config,
                    device=device,
                    skip_bertscore=args.skip_bertscore,
                )
                metrics["fold"] = fold.number
                fold_results["flan_finetuned"].append(metrics)
                training_history["flan_finetuned"][str(fold.number)] = history
                if args.save_private_errors:
                    _save_private_errors(
                        private_error_path,
                        fold=fold,
                        model_name="flan_finetuned",
                        predictions=predictions,
                    )
                del bundle

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        with (private_dir / "training_history.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(training_history, handle, indent=2, sort_keys=True)
        _write_results(
            public_dir,
            run_id=run_id,
            diagnostics=diagnostics,
            selected_folds=args.folds,
            fold_results=dict(fold_results),
            device=device,
            settings={
                "models": list(args.models),
                "prompt_variant": args.prompt_variant,
                "rationale_mode": args.rationale_mode,
                "skip_bertscore": bool(args.skip_bertscore),
                "reference_source": (
                    "local_silver_sidecar"
                    if args.reference_sidecar
                    else "provider_records"
                ),
                "seed": int(config["project"]["seed"]),
                "n_splits": int(config["project"]["n_splits"]),
                "protocol": "combined_stratified_cross_validation",
                "label_mapping": {
                    str(config["data"]["harmful_file"]): "harmful",
                    str(config["data"]["harmless_file"]): "harmless",
                    str(config["data"]["ambiguous_file"]): "unlabeled_stress_test",
                },
            },
        )

    print("Experiment finished. No raw record values were printed.")
    print(f"Aggregate results: {public_dir}")
    print(f"Private checkpoints and diagnostics: {private_dir}")
    return public_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the confidential DNLP experiment entirely offline."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=list(SUPPORTED_MODELS),
        help="Comma-separated: kimcnn,flan_zero,flan_finetuned",
    )
    parser.add_argument(
        "--folds",
        type=_parse_folds,
        default=[0, 1, 2, 3, 4],
        help="Comma-separated fold numbers.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=["full", "short"],
        default="full",
        help="Use 'short' only for the planned policy-prompt ablation.",
    )
    parser.add_argument(
        "--rationale-mode",
        choices=["gold", "generic"],
        default="gold",
        help="Use 'generic' only for the planned fine-tuning rationale ablation.",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Smoke tests only. Final reported runs must calculate BERTScore.",
    )
    parser.add_argument(
        "--reference-sidecar",
        help=(
            "Local JSON mapping opaque example IDs to refusal references. "
            "The file remains private and is never copied into public results."
        ),
    )
    parser.add_argument(
        "--save-private-errors",
        action="store_true",
        help="Save confidential misclassified text locally for manual analysis.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
