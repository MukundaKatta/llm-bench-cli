# llm-bench-cli

A fast, lightweight CLI tool for benchmarking local LLMs. Compare inference speed, throughput, and quality across models running on your own hardware.

## Why llm-bench-cli?

2026 is the year of efficient AI. With dozens of local LLM options (Ollama, llama.cpp, vLLM, etc.), choosing the right model for your hardware is harder than ever. This tool makes it easy to run standardized benchmarks and compare results side-by-side.

## Features

- Benchmark any local LLM accessible via an OpenAI-compatible API
- Measure tokens/second, time-to-first-token, and total latency
- Built-in test prompts for coding, reasoning, creative writing, and summarization
- Side-by-side comparison tables in your terminal
- Export results to JSON or CSV
- Supports concurrent request testing for throughput measurement
- Works with Ollama, LM Studio, vLLM, llama.cpp server, and any OpenAI-compatible endpoint

## Installation

```bash
pip install llm-bench-cli
```

Or install from source:

```bash
git clone https://github.com/MukundaKatta/llm-bench-cli.git
cd llm-bench-cli
pip install -e .
```

## Quick Start

```bash
# Benchmark a single model (assumes Ollama running on default port)
llm-bench run --model llama3.2 --endpoint http://localhost:11434/v1

# Compare two models
llm-bench compare --models llama3.2,mistral-7b --endpoint http://localhost:11434/v1

# Run with specific test categories
llm-bench run --model llama3.2 --categories coding,reasoning

# Export results to JSON
llm-bench run --model llama3.2 --output results.json

# Throughput test with concurrent requests
llm-bench throughput --model llama3.2 --concurrency 4 --requests 20
```

## Python API

```python
from llm_bench import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig(
    endpoint="http://localhost:11434/v1",
    model="llama3.2",
    categories=["coding", "reasoning"],
    num_runs=3,
)

runner = BenchmarkRunner(config)
results = runner.run()

results.print_summary()
results.to_json("results.json")
```

## Test Categories

- **coding** - Code generation and debugging tasks
- **reasoning** - Logic puzzles and math problems
- **creative** - Creative writing prompts
- **summarization** - Text summarization tasks
- **instruction** - Instruction following accuracy

## Output Example

```
Model: llama3.2 (3B) @ http://localhost:11434/v1
Runs: 3 | Categories: coding, reasoning
---------------------------------------------------
Metric                    Mean       P50        P95
---------------------------------------------------
Time to First Token (ms)  145.2     142.0      168.3
Tokens/Second             42.8      43.1       41.2
Total Latency (ms)        2341.5    2298.0     2520.1
Output Tokens             98.3      96.0       112.0
---------------------------------------------------
```

## Contributing

Contributions are welcome! Fork the repo, create a feature branch, and open a PR.

## License

MIT License - see [LICENSE](LICENSE) for details.
