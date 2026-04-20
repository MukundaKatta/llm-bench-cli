"""Normalized benchmark result models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Normalized result from one benchmark run."""

    provider: str
    model: str
    base_url: str
    prompt: str
    status_code: int
    success: bool
    latency_ms: float
    total_duration_ms: float
    ttft_ms: float | None = None
    output_text: str = ""
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None
    raw_metrics: dict = field(default_factory=dict)
