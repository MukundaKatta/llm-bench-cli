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

        response = client.post(url, json=payload, timeout=timeout)
        duration_ms = (time.perf_counter() - started) * 1000
        data = response.json()

        if response.is_success:
            usage = data.get("usage", {})
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

        with client.stream("POST", url, json=payload, timeout=timeout) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        continue
                    data = json.loads(chunk)
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    delta = (
                        data.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        content_parts.append(delta)
                    if "usage" in data:
                        usage = data["usage"]

            duration_ms = (time.perf_counter() - started) * 1000

            if response.is_success:
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
                error=response.read().decode("utf-8", errors="ignore"),
                raw_metrics={"stream": True},
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
