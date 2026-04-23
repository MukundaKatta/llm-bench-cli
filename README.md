# llm-bench-cli

`llm-bench-cli` is an early-stage project for benchmarking local LLMs and OpenAI-compatible inference endpoints, with a focus on throughput, latency, and side-by-side model comparison on your own hardware.

As local inference gets more practical, the hard part is often not running a model but choosing between several of them. Different models, runtimes, and quantizations can behave very differently on the same machine. `llm-bench-cli` is aimed at making those tradeoffs easier to measure.

## Why llm-bench-cli

The long-term goal is to provide a lightweight benchmarking workflow for developers who care about:

- tokens per second
- time to first token
- end-to-end latency
- throughput under concurrency
- practical comparison across local-model runtimes

## Current Status

`llm-bench-cli` is currently in an early public stage.

The repository already captures the project intent and basic dependency direction, but it does not yet contain the full CLI implementation described by the original concept. This README has been updated to reflect the current state honestly while keeping the intended direction clear.

## Project Direction

The project is intended to grow toward:

- a simple CLI for repeatable local-model benchmarks
- comparisons across models served via OpenAI-compatible APIs
- benchmark suites for coding, reasoning, summarization, and instruction-following tasks
- exportable benchmark results for tracking performance over time
- lightweight reporting that works well in terminals and scripts

## Planned Workflow

The intended user flow is straightforward:

1. Point the tool at a local or self-hosted inference endpoint.
2. Run a benchmark suite against one or more models.
3. Compare latency and throughput across runs.
4. Export results for later analysis.

## Why This Matters

For many builders, model choice is now a hardware and latency question as much as a quality question. A small benchmarking tool can help make those decisions more empirical, especially when comparing Ollama, llama.cpp servers, LM Studio, vLLM, and other OpenAI-compatible endpoints.

## Installation

Install from source:

```bash
git clone https://github.com/MukundaKatta/llm-bench-cli.git
cd llm-bench-cli
pip install -e .
```

## First Working Benchmark

The first runnable slice benchmarks an OpenAI-compatible `/v1/chat/completions` endpoint.

Required environment variables:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=test-key
```

Run a benchmark:

```bash
llm-bench bench \
  --model your-model-name \
  --prompt "Explain what makes a fast benchmark useful."
```

Optional flags:

- `--stream` to measure time to first token when the endpoint supports streaming
- `--json-output` to print the normalized result schema for scripts and later comparisons
- `--timeout` to raise or lower the request timeout

Example JSON result:

```json
{
  "provider": "openai-compatible",
  "model": "your-model-name",
  "base_url": "http://localhost:8000/v1",
  "prompt": "Explain what makes a fast benchmark useful.",
  "status_code": 200,
  "success": true,
  "latency_ms": 123.4,
  "total_duration_ms": 245.7,
  "ttft_ms": 123.4,
  "output_text": "A useful benchmark...",
  "output_tokens": 42,
  "prompt_tokens": 18,
  "total_tokens": 60,
  "error": null,
  "raw_metrics": {
    "stream": true
  }
}
```

## Current Repository Contents

```text
llm-bench-cli/
├── README.md
├── pyproject.toml
├── src/llm_bench_cli/
│   ├── adapters.py
│   ├── cli.py
│   ├── models.py
│   └── __init__.py
└── tests/
    └── test_cli.py
```

## Roadmap

Near-term priorities now include:

- adding more benchmark suites and prompts
- exporting run results to files
- supporting more provider adapters behind the same schema
- adding concurrency and throughput runs
- expanding reporting beyond a single benchmark invocation

## Contributing

Contributions, ideas, and feedback are welcome, especially around:

- benchmark design
- local-model comparison workflows
- CLI ergonomics
- reporting formats
- evaluation categories that are practical for developers

## License

MIT License. See [LICENSE](LICENSE) for details.
