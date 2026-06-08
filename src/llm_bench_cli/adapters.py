"""Provider adapters for benchmark backends."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .models import BenchmarkResult


class ProviderAdapter(ABC):
    """Adapter contract for benchmarking inference providers."""

    provider_name: str

    @abstractmethod
    def benchmark(
        self,
        *,
        client: httpx.Client,
        base_url: str,
        model: str,
        prompt: str,
        max_tokens: int,
        stream: bool,
        timeout: float,
    ) -> BenchmarkResult:
        """Execute one benchmark run and return a normalized result."""


class OpenAICompatibleAdapter(ProviderAdapter):
    """Benchmark an OpenAI-compatible /v1/chat/completions endpoint."""

    provider_name = "openai-compatible"

    def benchmark(
        self,
        *,
        client: httpx.Client,
        base_url: str,
        model: str,
        prompt: str,
        max_tokens: int,
        stream: bool,
        timeout: float,
    ) -> BenchmarkResult:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": stream,
        }
        url = base_url.rstrip("/") + "/chat/completions"
        started = time.perf_counter()

        if stream:
            return self._benchmark_streaming(
                client=client,
                url=url,
                payload=payload,
                started=started,
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout=timeout,
            )

        try:
            response = client.post(url, json=payload, timeout=timeout)
        except httpx.RequestError as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            return self._request_error_result(
                base_url=base_url,
                model=model,
                prompt=prompt,
                duration_ms=duration_ms,
                error=f"{type(exc).__name__}: {exc}",
                stream=False,
            )
        duration_ms = (time.perf_counter() - started) * 1000

        if not response.is_success:
            return BenchmarkResult(
                provider=self.provider_name,
                model=model,
                base_url=base_url,
                prompt=prompt,
                status_code=response.status_code,
                success=False,
                latency_ms=duration_ms,
                total_duration_ms=duration_ms,
                error=response.text,
                raw_metrics={"stream": False},
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            return BenchmarkResult(
                provider=self.provider_name,
                model=model,
                base_url=base_url,
                prompt=prompt,
                status_code=response.status_code,
                success=False,
                latency_ms=duration_ms,
                total_duration_ms=duration_ms,
                error=f"Invalid JSON in response body: {exc}",
                raw_metrics={"stream": False},
            )

        usage = data.get("usage") or {}
        content = self._extract_output_text(data)
        return BenchmarkResult(
            provider=self.provider_name,
            model=model,
            base_url=base_url,
            prompt=prompt,
            status_code=response.status_code,
            success=True,
            latency_ms=duration_ms,
            total_duration_ms=duration_ms,
            output_text=content,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            raw_metrics={"stream": False},
        )

    def _benchmark_streaming(
        self,
        *,
        client: httpx.Client,
        url: str,
        payload: dict[str, Any],
        started: float,
        base_url: str,
        model: str,
        prompt: str,
        timeout: float,
    ) -> BenchmarkResult:
        first_token_ms: float | None = None
        content_parts: list[str] = []
        usage: dict[str, Any] = {}

        try:
            stream_ctx = client.stream("POST", url, json=payload, timeout=timeout)
        except httpx.RequestError as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            return self._request_error_result(
                base_url=base_url,
                model=model,
                prompt=prompt,
                duration_ms=duration_ms,
                error=f"{type(exc).__name__}: {exc}",
                stream=True,
            )

        with stream_ctx as response:
            if not response.is_success:
                error_body = response.read().decode("utf-8", errors="ignore")
                duration_ms = (time.perf_counter() - started) * 1000
                return BenchmarkResult(
                    provider=self.provider_name,
                    model=model,
                    base_url=base_url,
                    prompt=prompt,
                    status_code=response.status_code,
                    success=False,
                    latency_ms=duration_ms,
                    total_duration_ms=duration_ms,
                    error=error_body,
                    raw_metrics={"stream": True},
                )

            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        chunk = line[6:].strip()
                        if not chunk or chunk == "[DONE]":
                            continue
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or [{}]
                        delta = (choices[0].get("delta") or {}).get("content", "")
                        if delta:
                            content_parts.append(delta)
                        if data.get("usage"):
                            usage = data["usage"]
            except httpx.RequestError as exc:
                duration_ms = (time.perf_counter() - started) * 1000
                return BenchmarkResult(
                    provider=self.provider_name,
                    model=model,
                    base_url=base_url,
                    prompt=prompt,
                    status_code=response.status_code,
                    success=False,
                    latency_ms=first_token_ms or duration_ms,
                    total_duration_ms=duration_ms,
                    ttft_ms=first_token_ms,
                    error=f"{type(exc).__name__}: {exc}",
                    raw_metrics={"stream": True},
                )

            duration_ms = (time.perf_counter() - started) * 1000

            return BenchmarkResult(
                provider=self.provider_name,
                model=model,
                base_url=base_url,
                prompt=prompt,
                status_code=response.status_code,
                success=True,
                latency_ms=first_token_ms or duration_ms,
                total_duration_ms=duration_ms,
                ttft_ms=first_token_ms,
                output_text="".join(content_parts),
                prompt_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                raw_metrics={"stream": True},
            )

    def _request_error_result(
        self,
        *,
        base_url: str,
        model: str,
        prompt: str,
        duration_ms: float,
        error: str,
        stream: bool,
    ) -> BenchmarkResult:
        """Build a failed result for a transport-level error (no HTTP response)."""
        return BenchmarkResult(
            provider=self.provider_name,
            model=model,
            base_url=base_url,
            prompt=prompt,
            status_code=0,
            success=False,
            latency_ms=duration_ms,
            total_duration_ms=duration_ms,
            error=error,
            raw_metrics={"stream": stream},
        )

    @staticmethod
    def _extract_output_text(data: dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
