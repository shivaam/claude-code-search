from claude_code_search.ollama_client import OllamaClient, OllamaError


class FakeTransport:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self._status = status
        self.last_request: dict | None = None

    def post(self, url: str, json_body: dict, timeout: float) -> tuple[int, dict]:
        self.last_request = {"url": url, "json": json_body, "timeout": timeout}
        return self._status, self._payload


def test_generate_returns_response_text() -> None:
    tp = FakeTransport({"response": "hello from ollama"})
    client = OllamaClient(url="http://localhost:11434", transport=tp)
    out = client.generate(model="gemma2:9b", prompt="Say hi")
    assert out == "hello from ollama"
    assert tp.last_request is not None
    assert tp.last_request["url"].endswith("/api/generate")
    assert tp.last_request["json"]["model"] == "gemma2:9b"
    assert tp.last_request["json"]["stream"] is False


def test_generate_raises_on_bad_status() -> None:
    tp = FakeTransport({"error": "model not found"}, status=404)
    client = OllamaClient(url="http://localhost:11434", transport=tp)
    try:
        client.generate(model="nope", prompt="Say hi")
    except OllamaError as e:
        assert "404" in str(e)
    else:
        raise AssertionError("expected OllamaError")
