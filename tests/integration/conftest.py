"""Pytest configuration for integration tests."""

import logging
import os
import shutil
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# Compose CLI executable (docker or podman).
_COMPOSE_CLI = shutil.which("docker") or shutil.which("podman") or "docker"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration tests unless RUN_INTEGRATION=1."""
    _ = config
    if os.getenv("RUN_INTEGRATION") == "1":
        return

    skip_marker = pytest.mark.skip(
        reason="Integration tests are disabled. Set RUN_INTEGRATION=1."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(autouse=True)  # type: ignore[misc]
def clean_data_directory() -> Generator[None]:
    """Clean the data directory before and after each integration test."""

    data_dir = Path("data")
    _clear_data_dir(data_dir)
    yield
    _clear_data_dir(data_dir)


def _clear_data_dir(data_dir: Path) -> None:
    """Remove all contents inside *data_dir*.

    Files written by the batch container may be owned by a different UID.
    ``shutil.rmtree`` will fail with ``PermissionError`` in that case.
    When that happens we fall back to running ``rm -rf`` inside the
    container which owns the files.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for child in data_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError:
            logger.info("Permission denied removing %s, delegating to container", child)
            _container_rm(data_dir)
            return


def _container_rm(data_dir: Path) -> None:
    """Remove data directory contents via the batch container."""
    result = subprocess.run(
        [
            _COMPOSE_CLI,
            "compose",
            "-f",
            "compose.yaml",
            "run",
            "--rm",
            "-T",
            "--entrypoint",
            "sh",
            "batch",
            "-c",
            f"rm -rf /app/{data_dir}/*",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(
            "Container cleanup failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip(),
        )
