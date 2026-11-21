from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path


def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# Outputs of the TRAIN stage in dvc.yaml.
TRAIN_OUTPUTS = (
    "models/model.pt",
    "reports/run_manifest.json",
    "reports/train_summary.json",
    "reports/training_curves.png",
)


def _git(*args: str) -> str | None:
    """Run a Git command and return its output, or None if Git fails
    (for example when the folder is not a Git repository)."""
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
    except OSError:  # Git is not installed
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata: code, data, environment,
    settings and MLflow Run ID used by one training Run."""
    import torch  # imported here so the module stays light for other callers

    commit = _git("rev-parse", "HEAD")
    # DVC deletes TRAIN's outputs before running it, and some of them are
    # tracked by Git, so leave them out: only code/config changes count.
    excluded = [f":(exclude){path}" for path in TRAIN_OUTPUTS]
    status = _git("status", "--porcelain", "--", ".", *excluded)
    train = params["train"]
    augmentation = params["augmentation"]

    return {
        "git": {
            "commit": commit or "unavailable",
            # None means Git could not tell us (not a repository).
            "dirty": None if status is None else bool(status),
        },
        "data": {
            "training_data_identity": training_data_identity(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "dvc": _package_version("dvc"),
            "mlflow": _package_version("mlflow"),
            "cuda_runtime": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "execution": {
            "device": str(device),
            "random_state": int(train["random_state"]),
            "mlflow_run_id": mlflow_run_id,
        },
        "configuration": {
            "image_size": int(params["data"]["image_size"]),
            "batch_size": int(train["batch_size"]),
            "epochs": int(train["epochs"]),
            "learning_rate": float(train["learning_rate"]),
            "dropout": float(train["dropout"]),
            "augmentation_enabled": bool(augmentation["enabled"]),
            "horizontal_flip": float(augmentation["horizontal_flip"]),
            "rotation_degrees": float(augmentation["rotation_degrees"]),
        },
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
