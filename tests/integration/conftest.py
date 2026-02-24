"""Pytest configuration for integration tests."""

import os
import shutil
import stat
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest


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


def _handle_remove_readonly(
    func: Callable[..., Any], path: str, exc: BaseException
) -> object:
    """Error handler for shutil.rmtree when files are read-only."""
    if stat.S_ISREG(os.stat(path).st_mode):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)
        return None
    else:
        raise exc


def _clear_data_dir(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for child in data_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, onexc=_handle_remove_readonly)
        else:
            child.unlink(missing_ok=True)
