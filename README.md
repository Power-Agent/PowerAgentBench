# PowerAgentBench

PowerAgentBench is a benchmark suite for evaluating AI agents on power-system operation and planning tasks. The current release focuses on steady-state studies and includes conventional scripted baselines, Ollama-hosted LLM agents, and OpenAI/ChatGPT-style LLM agents.

The benchmark is built around a public/hidden split. Agents see public case data, action spaces, and tool APIs. A hidden evaluator recomputes physical validity and returns discovery, evidence, safety, mitigation, efficiency, workflow, and reliability metrics.

## Repository Structure

```text
PowerAgentBench/
├── cases/                                      # Network case data in multiple formats
│   └── case39/
│       ├── pypsa/case39.nc                     # PyPSA netCDF format
│       ├── matpower/case39.m                   # MATPOWER .m format
│       └── pandapower/case39.json              # PandaPower JSON format
├── benchmarks/                                 # Benchmark definitions and task configs
│   └── steady/
│       ├── level_1/                            # N-1 steady-state audit and mitigation
│       │   ├── README.md                       # Full Level 1 benchmark specification
│       │   ├── actionspace.json                # Action contract and operating limits
│       │   ├── actioncost.json                 # Per-step action costs
│       │   ├── baseline_summary.json
│       │   └── solution_template.json
│       └── level_2/                            # Agentic N-2 search and mitigation
│           ├── README.md                       # Full Level 2 benchmark specification
│           ├── .env.example                    # Template for private model/API configuration
│           ├── .gitignore                      # Keeps local .env files out of git
│           └── prompts/
│               └── steady_n2_llm_prompt.json   # Shared LLM tool-use prompt template
├── scripts/                                    # Runnable entry points
│   ├── build_case.py                           # Rebuild the stressed Level 1 scenario
│   ├── convert_case.py                         # Export case39 to MATPOWER and PandaPower
│   ├── evaluate_solution.py                    # Score a Level 1 solution
│   ├── run_steady_n2_baselines.py              # Run Level 2 scripted baselines
│   ├── run_steady_n2_ollama_eval.py            # Run Level 2 Ollama-hosted LLM agents
│   └── run_steady_n2_openai_eval.py            # Run Level 2 OpenAI/ChatGPT-style agents
└── poweragentbench/                            # Shared library code
    ├── benchmark_utils.py                      # Level 1 case construction and scoring
    ├── steady_state_agentic.py                 # Level 2 DC N-2 evaluator and baselines
    ├── llm_agent_adapter.py                    # Provider-agnostic JSON-command LLM adapter
    ├── ollama_client.py                        # Ollama generate/chat client
    └── openai_client.py                        # OpenAI Responses API client
```

## Installation

```bash
pip install -e .
```

The package intentionally uses lightweight Python dependencies. Provider SDKs are not required for the built-in Ollama and OpenAI runners because both clients use standard-library HTTP calls.

## Quick Start

### Level 1: N-1 steady-state audit and mitigation

```bash
# Rebuild the benchmark case from source
python scripts/build_case.py

# Export to MATPOWER and PandaPower formats
python scripts/convert_case.py

# Evaluate a solution
python scripts/evaluate_solution.py \
  --solution benchmarks/steady/level_1/solution_template.json
```

### Level 2: Agentic N-2 search and mitigation

Run scripted baselines on deterministic variants of the existing IEEE 39-bus case:

```bash
python scripts/run_steady_n2_baselines.py \
  --case-source case39 \
  --cases 8 \
  --budget 80 \
  --report-k 20
```

Run deployed Ollama LLM agents:

```bash
python scripts/run_steady_n2_ollama_eval.py \
  --case-source case39 \
  --cases 8 \
  --budget 80 \
  --report-k 20 \
  --max-turns 12 \
  --prompt-template benchmarks/steady/level_2/prompts/steady_n2_llm_prompt.json
```

Run an OpenAI/ChatGPT-style agent, for example GPT-5.5:

```bash
python scripts/run_steady_n2_openai_eval.py \
  --case-source case39 \
  --cases 8 \
  --budget 80 \
  --report-k 20 \
  --max-turns 12 \
  --prompt-template benchmarks/steady/level_2/prompts/steady_n2_llm_prompt.json
```

Outputs are written under `results/steady_n2/` for Ollama runs and `results/steady_n2_openai/` for OpenAI runs. Each run produces per-case CSVs, aggregate CSVs, tool logs, sanitized API debug files, and LaTeX table rows.

## Case Formats

The IEEE 39-bus stressed scenario is provided in three formats so that agents and solvers are not tied to a single tool:

- **PyPSA** (`cases/case39/pypsa/case39.nc`): primary format used by the Level 1 evaluator and by the Level 2 case39 converter.
- **PandaPower** (`cases/case39/pandapower/case39.json`): for PandaPower-based tools.
- **MATPOWER** (`cases/case39/matpower/case39.m`): for MATPOWER or MATPOWER-compatible solvers.

## Benchmarks

### Steady Level 1

`benchmarks/steady/level_1/` evaluates N-1 steady-state audit and mitigation on a stressed IEEE 39-bus case. The agent receives a case, a published contingency list, and a bounded action space. The evaluator checks base-case and contingency violations after the submitted actions.

See:

```text
benchmarks/steady/level_1/README.md
```

### Steady Level 2

`benchmarks/steady/level_2/` evaluates agentic N-2 contingency search and optional mitigation. The agent must spend a limited validation budget, submit evidence-backed ranked contingencies, and optionally improve the hidden post-action violation score.

The default case source is the existing IEEE 39-bus case distributed in this repository. The runner converts it to a lightweight DC representation and creates deterministic operating-point variants from fixed seeds. A synthetic fallback is also available for development.

See:

```text
benchmarks/steady/level_2/README.md
```

## Model and API Configuration

Private model endpoints and API keys should not be committed to the repository. Configure them through a local `.env` file:

```bash
cp benchmarks/steady/level_2/.env.example benchmarks/steady/level_2/.env
```

The local `.env` file is ignored by Git. You may also pass the same settings through command-line flags or process environment variables.

### Ollama configuration

Example local Ollama settings:

```bash
POWERAGENTBENCH_OLLAMA_URL=http://localhost:11434/api/generate
POWERAGENTBENCH_OLLAMA_MODELS=qwen3.5:latest mistral-nemo:12b command-r:35b
POWERAGENTBENCH_OLLAMA_TEMPERATURE=0.0
POWERAGENTBENCH_OLLAMA_NUM_CTX=16384
POWERAGENTBENCH_OLLAMA_API_MODE=generate
POWERAGENTBENCH_OLLAMA_THINK=false
POWERAGENTBENCH_OLLAMA_SCHEMA_FORMAT=true
```

For internal deployments, replace `POWERAGENTBENCH_OLLAMA_URL` locally. Do not commit internal URLs.

Some Ollama models expose a `thinking` field when `POWERAGENTBENCH_OLLAMA_THINK=true`. PowerAgentBench treats this only as a generation option. Raw thinking traces are not parsed, scored, or required for benchmark results.

### OpenAI/ChatGPT configuration

Example local OpenAI settings:

```bash
POWERAGENTBENCH_OPENAI_API_KEY=sk-your-private-token
POWERAGENTBENCH_OPENAI_MODELS=gpt-5.5
POWERAGENTBENCH_OPENAI_URL=https://api.openai.com/v1/responses
POWERAGENTBENCH_OPENAI_TEMPERATURE=none
POWERAGENTBENCH_OPENAI_MAX_OUTPUT_TOKENS=4096
POWERAGENTBENCH_OPENAI_STRUCTURED_OUTPUTS=true
POWERAGENTBENCH_OPENAI_REASONING_EFFORT=medium
POWERAGENTBENCH_OPENAI_REASONING_SUMMARY=none
POWERAGENTBENCH_OPENAI_TIMEOUT=300
POWERAGENTBENCH_OPENAI_MAX_RETRIES=3
POWERAGENTBENCH_OPENAI_RETRY_BACKOFF=2.0
```

Many reasoning models reject a `temperature` parameter. Use `POWERAGENTBENCH_OPENAI_TEMPERATURE=none` to omit it. The OpenAI runner uses sanitized API debug logs and does not store the API key, raw output text, or reasoning content.

If a run times out, increase the timeout:

```bash
python scripts/run_steady_n2_openai_eval.py \
  --case-source case39 \
  --cases 8 \
  --budget 80 \
  --report-k 20 \
  --max-turns 12 \
  --prompt-template benchmarks/steady/level_2/prompts/steady_n2_llm_prompt.json \
  --timeout 600
```

## Metrics

PowerAgentBench returns per-case and aggregate metrics, including:

- submitted, evidence-backed, and found top-20 recall,
- evidence rate and unvalidated-claim rate,
- best severity capture and severity regret,
- false-safe rates and severity-weighted false negatives,
- post-action violation and violation reduction,
- action cost,
- invalid tool calls,
- schema repairs and type coercions,
- duplicate validation requests,
- explicit submission and auto-finalization indicators,
- validation budget use,
- completed and requested case counts.

These metrics distinguish answer quality, tool evidence, search quality, mitigation quality, safety behavior, and workflow compliance.

## Result Files

Typical Level 2 baseline outputs:

```text
results/steady_n2/baseline_per_case.csv
results/steady_n2/baseline_summary.csv
```

Typical Ollama outputs:

```text
results/steady_n2/ollama_all_per_case.csv
results/steady_n2/ollama_all_summary.csv
results/steady_n2/<model>_per_case.csv
results/steady_n2/<model>_summary.csv
results/steady_n2/<model>_tool_logs.jsonl
results/steady_n2/<model>_api_debug.jsonl
```

Typical OpenAI outputs:

```text
results/steady_n2_openai/openai_all_per_case.csv
results/steady_n2_openai/openai_all_summary.csv
results/steady_n2_openai/<model>-OpenAI_per_case.csv
results/steady_n2_openai/<model>-OpenAI_summary.csv
results/steady_n2_openai/<model>-OpenAI_tool_logs.jsonl
results/steady_n2_openai/<model>-OpenAI_api_debug.jsonl
```

If an OpenAI run stops early after a retry failure, partial outputs are preserved with `_partial` in the filename and errors are written to an errors JSONL file.

## Development Notes

- Use Level 1 to test basic steady-state action submission and physical validation.
- Use Level 2 to test agentic behavior, tool use, validation-budget allocation, evidence-backed reporting, and LLM workflows.
- Keep hidden oracle quantities, private endpoint URLs, and API keys outside the public repository.
- Rotate any API key that is accidentally shared or committed.
- Regenerate results after modifying prompts, adapters, scoring rules, or case-generation settings.
