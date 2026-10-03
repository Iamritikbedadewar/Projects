from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnlp_refusal.config import load_config, project_root
from dnlp_refusal.data import inspect_json_structure
from dnlp_refusal.privacy import (
    NetworkBlocker,
    enable_offline_environment,
    validate_local_data_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect JSON structure without printing any record values."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    enable_offline_environment()
    config = load_config(args.config)
    data_dir = validate_local_data_dir(
        args.data_dir,
        reject_cloud_synced_paths=bool(
            config["privacy"].get("reject_cloud_synced_paths", True)
        ),
    )

    filenames = [
        config["data"]["harmful_file"],
        config["data"]["harmless_file"],
        config["data"]["ambiguous_file"],
    ]
    reports = []
    with NetworkBlocker(enabled=True):
        for filename in filenames:
            path = data_dir / str(filename)
            if path.is_file():
                reports.append(inspect_json_structure(path))
            else:
                reports.append({"file": path.name, "missing": True})

    print(json.dumps(reports, indent=2, sort_keys=True))
    print("Structure only: no record values were printed.")


if __name__ == "__main__":
    main()

