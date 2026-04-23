"""CLI entrypoint for llm-bench-cli."""

from __future__ import annotations

import json
from dataclasses import asdict

import click
import httpx
from rich.console import Console
from rich.table import Table

from .adapters import OpenAICompatibleAdapter

console = Console()


@click.group()
def main() -> None:
    """Benchmark OpenAI-compatible inference endpoints."""


@main.command("bench")
@click.option("--base-url", envvar="OPENAI_BASE_URL", required=True, help="Base URL ending in /v1.")
@click.option("--api-key", envvar="OPENAI_API_KEY", default="test-key", show_default=True)
@click.option("--model", required=True, help="Model name to benchmark.")
@click.option("--prompt", required=True, help="Prompt to send.")
@click.option("--max-tokens", default=128, show_default=True, type=int)
@click.option("--stream/--no-stream", default=False, show_default=True)
@click.option("--timeout", default=60.0, show_default=True, type=float)
@click.option("--json-output", is_flag=True, help="Print normalized result JSON.")
def bench(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    timeout: float,
    json_output: bool,
) -> None:
    """Run one benchmark against an OpenAI-compatible endpoint."""
    adapter = OpenAICompatibleAdapter()
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(headers=headers) as client:
        result = adapter.benchmark(
            client=client,
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream,
            timeout=timeout,
        )

    if json_output:
        click.echo(json.dumps(asdict(result), indent=2))
        raise SystemExit(0 if result.success else 1)

    table = Table(title="Benchmark Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Provider", result.provider)
    table.add_row("Model", result.model)
    table.add_row("Base URL", result.base_url)
    table.add_row("Status", "success" if result.success else "failed")
    table.add_row("HTTP status", str(result.status_code))
    table.add_row("Latency (ms)", f"{result.latency_ms:.2f}")
    table.add_row("Total duration (ms)", f"{result.total_duration_ms:.2f}")
    table.add_row("TTFT (ms)", f"{result.ttft_ms:.2f}" if result.ttft_ms is not None else "n/a")
    table.add_row("Prompt tokens", str(result.prompt_tokens) if result.prompt_tokens is not None else "n/a")
    table.add_row("Output tokens", str(result.output_tokens) if result.output_tokens is not None else "n/a")
    table.add_row("Total tokens", str(result.total_tokens) if result.total_tokens is not None else "n/a")
    if result.error:
        table.add_row("Error", result.error)
    console.print(table)

    if result.success:
        preview = result.output_text.strip().replace("\n", " ")
        if preview:
            console.print(f"\n[bold]Output preview:[/] {preview[:200]}")

    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
