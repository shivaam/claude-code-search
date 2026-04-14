"""Minimal HTTP client for the local ollama server."""
from __future__ import annotations

from typing import Protocol


class OllamaError(RuntimeError):
    pass


class Transport(Protocol):
    def post(
        self, url: str, json_body: dict, timeout: float
    ) -> tuple[int, dict]: ...


class _HttpxTransport:
    def __init__(self) -> None:
        import httpx

        self._httpx = httpx

    def post(
        self, url: str, json_body: dict, timeout: float
    ) -> tuple[int, dict]:
        try:
            r = self._httpx.post(url, json=json_body, timeout=timeout)
        except self._httpx.ConnectError:
            raise OllamaError(
                f"cannot connect to ollama at {url}. "
                f"Is ollama running? Start it with: ollama serve"
            )
        except self._httpx.TimeoutException:
            raise OllamaError(
                f"ollama timed out after {timeout}s on {url}. "
                f"The model may still be loading — try again."
            )
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        return r.status_code, data


class OllamaClient:
    def __init__(
        self,
        url: str = "http://localhost:11434",
        transport: Transport | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.transport = transport or _HttpxTransport()
        self.timeout = timeout

    def generate(
        self, model: str, prompt: str, options: dict | None = None
    ) -> str:
        endpoint = f"{self.url}/api/generate"
        body = {"model": model, "prompt": prompt, "stream": False}
        if options:
            body["options"] = options
        status, data = self.transport.post(endpoint, body, self.timeout)
        if status != 200:
            raise OllamaError(f"ollama {endpoint} returned {status}: {data}")
        return str(data.get("response", "")).strip()
