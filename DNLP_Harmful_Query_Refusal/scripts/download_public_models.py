from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PUBLIC_MODELS = {
    "flan-t5-small": {
        "repo_id": "google/flan-t5-small",
        "local_dir": "models/flan-t5-small",
        "purpose": "zero-shot and locally fine-tuned classification/generation",
    },
    "bert-score-roberta-large": {
        "repo_id": "FacebookAI/roberta-large",
        "local_dir": "models/bert-score-roberta-large",
        "purpose": "standard English RoBERTa-large BERTScore evaluation at layer 17",
    },
}

PYTORCH_MODEL_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download public model weights only. This script has no data-directory "
            "argument and never opens private project data."
        )
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Requested Hub revision. The resolved commit SHA is recorded.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    api = HfApi()
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "notice": "Public model files only; no private dataset was opened.",
        "models": {},
    }

    for name, specification in PUBLIC_MODELS.items():
        repo_id = specification["repo_id"]
        print(f"Resolving public model: {repo_id}")
        info = api.model_info(repo_id, revision=args.revision)
        resolved_revision = info.sha
        if not resolved_revision:
            raise RuntimeError(f"Could not resolve a commit SHA for {repo_id}.")
        local_dir = root / specification["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=resolved_revision,
            local_dir=local_dir,
            allow_patterns=PYTORCH_MODEL_PATTERNS,
        )
        manifest["models"][name] = {
            "repo_id": repo_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "local_dir": specification["local_dir"],
            "purpose": specification["purpose"],
        }

    manifest_path = root / "models" / "MODEL_MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"Public models downloaded. Manifest: {manifest_path}")
    print("You may now disconnect the computer and run the offline experiment.")


if __name__ == "__main__":
    main()
