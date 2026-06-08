from __future__ import annotations

import json

import httpx
from click.testing import CliRunner

from llm_bench_cli.adapters import OpenAICompatibleAdapter
from llm_bench_cli.cli import main


def test_openai_adapter_non_streaming_normalizes_result():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        payload = {
            "choices": [{"message": {"content": "hello from benchmark"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            },
        }
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAdapter()

    result = adapter.benchmark(
        client=client,
        base_url="http://localhost:8000/v1",
        model="test-model",
        prompt="hello",
        max_tokens=32,
        stream=False,
        timeout=10.0,
    )

    assert result.success is True
    assert result.provider == "openai-compatible"
    assert result.output_text == "hello from benchmark"
    assert result.total_tokens == 16


def test_openai_adapter_streaming_records_ttft():
    body = (
        'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAdapter()

    result = adapter.benchmark(
        client=client,
        base_url="http://localhost:8000/v1",
        model="test-model",
        prompt="hello",
        max_tokens=32,
        stream=True,
        timeout=10.0,
    )

    assert result.success is True
    assert result.ttft_ms is not None
    assert result.output_text == "hello world"


def test_openai_adapter_error_status_non_json_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502, text="Bad Gateway", headers={"content-type": "text/plain"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAdapter()

    result = adapter.benchmark(
        client=client,
        base_url="http://localhost:8000/v1",
        model="test-model",
        prompt="hello",
        max_tokens=32,
        stream=False,
        timeout=10.0,
    )

    assert result.success is False
    assert result.status_code == 502
    assert "Bad Gateway" in result.error


def test_openai_adapter_success_invalid_json_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="not json", headers={"content-type": "text/plain"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAdapter()

    result = adapter.benchmark(
        client=client,
        base_url="http://localhost:8000/v1",
        model="test-model",
        prompt="hello",
        max_tokens=32,
        stream=False,
        timeout=10.0,
    )

    assert result.success is False
    assert result.status_code == 200
    assert "Invalid JSON" in result.error


def test_openai_adapter_request_error_does_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAdapter()

    result = adapter.benchmark(
        client=client,
        base_url="http://localhost:8000/v1",
        model="test-model",
        prompt="hello",
        max_tokens=32,
        stream=False,
        timeout=10.0,
    )

    assert result.success is False
    assert result.status_code == 0
    assert "ConnectError" in result.error


def test_openai_adapter_streaming_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, text="upstream model crashed", headers={"content-type": "text/plain"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAdapter()

    result = adapter.benchmark(
        client=client,
        base_url="http://localhost:8000/v1",
        model="test-model",
        prompt="hello",
        max_tokens=32,
        stream=True,
        timeout=10.0,
    )

    assert result.success is False
    assert result.status_code == 500
    assert "upstream model crashed" in result.error


def test_openai_adapter_streaming_skips_malformed_chunks():
    body = (
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        "data: {not valid json}\n\n"
        'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAdapter()

    result = adapter.benchmark(
        client=client,
        base_url="http://localhost:8000/v1",
        model="test-model",
        prompt="hello",
        max_tokens=32,
        stream=True,
        timeout=10.0,
    )

    assert result.success is True
    assert result.output_text == "ab"


def test_bench_command_supports_json_output(monkeypatch):
    def fake_benchmark(**_: object):
        from llm_bench_cli.models import BenchmarkResult

        return BenchmarkResult(
            provider="openai-compatible",
            model="test-model",
            base_url="http://localhost:8000/v1",
            prompt="hello",
            status_code=200,
            success=True,
            latency_ms=12.0,
            total_duration_ms=20.0,
            ttft_ms=12.0,
            output_text="ok",
            prompt_tokens=8,
            output_tokens=4,
            total_tokens=12,
        )

    monkeypatch.setattr(
        OpenAICompatibleAdapter, "benchmark", staticmethod(fake_benchmark)
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bench",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "test-model",
            "--prompt",
            "hello",
            "--json-output",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["provider"] == "openai-compatible"
    assert payload["total_tokens"] == 12
