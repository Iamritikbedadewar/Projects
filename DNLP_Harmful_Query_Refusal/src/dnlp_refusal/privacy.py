from __future__ import annotations

import os
import socket
from pathlib import Path
from types import TracebackType
from typing import Callable


class PrivacyError(RuntimeError):
    pass


CLOUD_PATH_MARKERS = (
    "onedrive",
    "google drive",
    "googledrive",
    "dropbox",
    "icloud",
    "box sync",
    "sharepoint",
)


def enable_offline_environment() -> None:
    """Disable supported telemetry and remote model lookups."""

    values = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in values.items():
        os.environ[key] = value


def validate_local_data_dir(
    value: str | Path,
    *,
    reject_cloud_synced_paths: bool = True,
) -> Path:
    """Resolve a local data directory without opening any dataset file."""

    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise PrivacyError("The supplied data directory does not exist or is not a folder.")

    lowered = str(path).casefold()
    if reject_cloud_synced_paths and any(
        marker in lowered for marker in CLOUD_PATH_MARKERS
    ):
        raise PrivacyError(
            "The data path appears to be inside a cloud-synchronized directory. "
            "Move it to a non-synchronized local folder first."
        )

    if (path / ".git").exists():
        raise PrivacyError(
            "The private data directory appears to be a Git repository. "
            "Use a separate non-versioned local folder."
        )
    return path


class NetworkBlocker:
    """Block outbound sockets for the duration of the experiment process."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._original_connect: Callable[..., object] | None = None
        self._original_connect_ex: Callable[..., object] | None = None
        self._original_create_connection: Callable[..., object] | None = None

    @staticmethod
    def _deny(*_args: object, **_kwargs: object) -> None:
        raise PrivacyError(
            "Outbound network access is blocked while confidential data may be in memory."
        )

    @staticmethod
    def _deny_ex(*_args: object, **_kwargs: object) -> int:
        raise PrivacyError(
            "Outbound network access is blocked while confidential data may be in memory."
        )

    def __enter__(self) -> "NetworkBlocker":
        if not self.enabled:
            return self
        self._original_connect = socket.socket.connect
        self._original_connect_ex = socket.socket.connect_ex
        self._original_create_connection = socket.create_connection
        socket.socket.connect = self._deny  # type: ignore[method-assign]
        socket.socket.connect_ex = self._deny_ex  # type: ignore[method-assign]
        socket.create_connection = self._deny  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self.enabled:
            return
        if self._original_connect is not None:
            socket.socket.connect = self._original_connect  # type: ignore[method-assign]
        if self._original_connect_ex is not None:
            socket.socket.connect_ex = self._original_connect_ex  # type: ignore[method-assign]
        if self._original_create_connection is not None:
            socket.create_connection = self._original_create_connection  # type: ignore[assignment]

