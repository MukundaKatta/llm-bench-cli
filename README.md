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

At the moment, the repository is still taking shape. For now, clone the repo to follow progress or contribute as the implementation develops:

```bash
git clone https://github.com/MukundaKatta/llm-bench-cli.git
cd llm-bench-cli
```

## Current Repository Contents

```text
llm-bench-cli/
├── README.md
├── requirements.txt
└── LICENSE
```

Current dependencies suggest the planned implementation direction:

- `httpx` for calling benchmark targets
- `rich` for terminal output
- `click` for the CLI interface

## Roadmap

Near-term priorities include:

- implementing the first runnable CLI commands
- defining benchmark task categories
- adding result formatting and export support
- documenting supported endpoint types
- creating reproducible example benchmark runs

## Contributing

Contributions, ideas, and feedback are welcome, especially around:

- benchmark design
- local-model comparison workflows
- CLI ergonomics
- reporting formats
- evaluation categories that are practical for developers

## License

MIT License. See [LICENSE](LICENSE) for details.
