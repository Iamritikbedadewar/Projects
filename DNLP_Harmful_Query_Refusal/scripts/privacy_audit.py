from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_NAMES = (
    "synthetic_nlq_abuse.json",
    "synthetic_nlq_ambiguous.json",
    "synthetic_nlq_mongo_pairs.json",
    "material_schema.json",
    "example-materials",
    "matplus-data",
    "private_error_analysis.csv",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "openai",
    "anthropic",
    "google.cloud",
    "boto3",
    "requests",
    "httpx",
    "wandb",
)
ALLOWED_NETWORK_FILE = ROOT / "scripts" / "download_public_models.py"


def source_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.py")
        if ".venv" not in path.parts
        and "private_outputs" not in path.parts
        and not path.is_relative_to(ROOT / "models")
    )


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def main() -> None:
    problems: list[str] = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        lowered = str(relative).casefold()
        if any(name.casefold() in lowered for name in SENSITIVE_NAMES):
            if relative.as_posix() in {
                "README.md",
                "PRIVACY.md",
                "SAFE_TO_SHARE.md",
                ".gitignore",
                "configs/default.yaml",
                "scripts/privacy_audit.py",
            }:
                continue
            problems.append(f"Sensitive-looking file present: {relative}")

    for path in source_files():
        if path == ALLOWED_NETWORK_FILE:
            continue
        for module in imported_modules(path):
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in FORBIDDEN_IMPORT_PREFIXES
            ):
                problems.append(
                    f"Forbidden network/API import in {path.relative_to(ROOT)}: {module}"
                )

    if (ROOT / "private_outputs").exists():
        private_files = [
            path
            for path in (ROOT / "private_outputs").rglob("*")
            if path.is_file()
        ]
        if private_files:
            problems.append(
                "private_outputs contains files. Exclude that directory from any submission."
            )

    if problems:
        print("PRIVACY AUDIT FAILED")
        for problem in problems:
            print(f"- {problem}")
        sys.exit(1)

    print("PRIVACY AUDIT PASSED")
    print("No provider data or prohibited external-API imports were found.")
    print("A manual agreement review is still required before sharing.")


if __name__ == "__main__":
    main()
