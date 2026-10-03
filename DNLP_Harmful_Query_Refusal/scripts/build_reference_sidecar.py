from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dnlp_refusal.config import load_config, project_root
from dnlp_refusal.data import load_project_dataset
from dnlp_refusal.privacy import (
    NetworkBlocker,
    enable_offline_environment,
    validate_local_data_dir,
)
from dnlp_refusal.prompts import generic_refusal


enable_offline_environment()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create private, rule-based silver refusal references offline."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = validate_local_data_dir(
        args.data_dir,
        reject_cloud_synced_paths=bool(
            config["privacy"].get("reject_cloud_synced_paths", True)
        ),
    )
    output_path = Path(args.output).expanduser().resolve()
    private_root = (project_root() / "private_outputs").resolve()
    if output_path != private_root and private_root not in output_path.parents:
        raise ValueError(
            "The reference sidecar must be stored inside private_outputs."
        )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing sidecar: {output_path.name}"
        )

    with NetworkBlocker(
        enabled=bool(config["privacy"].get("block_network", True))
    ):
        examples, diagnostics = load_project_dataset(
            data_dir,
            config["data"],
            require_harmful_references=False,
        )
        references = {
            example.uid: generic_refusal(example.text)
            for example in examples
            if example.label == "harmful"
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "method": "deterministic_policy_template_silver_references",
                    "notice": (
                        "Automatically generated local silver references, not "
                        "provider-supplied or human-authored gold references."
                    ),
                    "harmful_count": diagnostics.harmful_count,
                    "references": references,
                },
                handle,
                indent=2,
                sort_keys=True,
            )

    print(
        f"Created {len(references)} private silver references at {output_path}"
    )
    print("No source query text was written to the sidecar or printed.")


if __name__ == "__main__":
    main()
