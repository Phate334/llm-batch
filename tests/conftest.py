# mypy: disable-error-code=misc

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest


class DummyHeaders:
    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._values: dict[str, str] = {}
        if initial:
            for key, value in initial.items():
                self._values[key.lower()] = value

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._values.get(key.lower(), default)

    def set(self, key: str, value: str) -> None:
        self._values[key.lower()] = value


@dataclass
class DummyRequest:
    method: str = "POST"
    path: str = "/chat/completions"
    body_text: str = ""
    headers: DummyHeaders = field(default_factory=DummyHeaders)

    def get_text(self, strict: bool = False) -> str:
        return self.body_text


@dataclass
class DummyResponse:
    body_text: str = ""
    headers: DummyHeaders = field(default_factory=DummyHeaders)

    def get_text(self, strict: bool = False) -> str:
        return self.body_text


@dataclass
class DummyFlow:
    request: DummyRequest
    response: DummyResponse
    metadata: dict[str, object] = field(default_factory=dict)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def make_flow() -> Callable[..., DummyFlow]:
    def _make_flow(
        *,
        method: str = "POST",
        path: str = "/chat/completions",
        request_body: str = "",
        request_headers: dict[str, str] | None = None,
        response_body: str = "",
        response_headers: dict[str, str] | None = None,
    ) -> DummyFlow:
        request = DummyRequest(
            method=method,
            path=path,
            body_text=request_body,
            headers=DummyHeaders(request_headers),
        )
        response = DummyResponse(
            body_text=response_body,
            headers=DummyHeaders(response_headers),
        )
        return DummyFlow(request=request, response=response)

    return _make_flow
