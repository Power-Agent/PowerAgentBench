# PowerAgentBench

PowerAgentBench is a benchmark suite for evaluating AI agents on power system operational and planning tasks. The current release focuses on steady-state studies and includes both conventional scripted baselines and LLM/tool-agent evaluation.

The benchmark is designed around a public/hidden split. Agents see public case data, action spaces, and tool APIs. A hidden evaluator recomputes physical validity and returns discovery, evidence, mitigation, efficiency, and reliability metrics.

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
│           ├── .env.example                    # Template for private Ollama configuration
│           └── prompts/
│               └── steady_n2_llm_prompt.json   # Shared LLM tool-use prompt template
├── scripts/                                    # Runnable entry points
│   ├── build_case.py                           # Rebuild the stressed Level 1 scenario
│   ├── convert_case.py                         # Export case39 to MATPOWER and PandaPower
│   ├── evaluate_solution.py                    # Score a Level 1 solution
│   ├── run_steady_n2_baselines.py              # Run Level 2 scripted baselines
│   └── run_steady_n2_ollama_eval.py            # Run Level 2 Ollama-hosted LLM agents
└── poweragentbench/                            # Shared library code
    ├── benchmark_utils.py                      # Level 1 case construction and scoring
    ├── steady_state_agentic.py                 # Level 2 DC N-2 evaluator and baselines
    ├── llm_agent_adapter.py                    # JSON-command LLM adapter
    └── ollama_client.py                        # Ollama generate/chat client
```

## Installation

```bash
pip install -e .
```

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

Outputs are written under `results/steady_n2/` as per-case CSVs, aggregate CSVs, tool logs, API debug files, and LaTeX table rows.

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

## Ollama Configuration

Private or internal Ollama endpoints should not be committed to the repository. Configure them through a local `.env` file.

```bash
cp benchmarks/steady/level_2/.env.example benchmarks/steady/level_2/.env
```

Example local settings:

```bash
POWERAGENTBENCH_OLLAMA_URL=http://localhost:11434/api/generate
POWERAGENTBENCH_OLLAMA_MODELS=qwen3.5:latest mistral-nemo:12b command-r:35b
POWERAGENTBENCH_OLLAMA_TEMPERATURE=0.0
POWERAGENTBENCH_OLLAMA_NUM_CTX=16384
POWERAGENTBENCH_OLLAMA_API_MODE=generate
POWERAGENTBENCH_OLLAMA_THINK=false
POWERAGENTBENCH_OLLAMA_SCHEMA_FORMAT=true
```

The local `.env` file is ignored by Git. You may also pass the same settings through command-line flags or process environment variables.

## Metrics

PowerAgentBench returns per-case and aggregate metrics, including:

- submitted and evidence-backed top-20 recall,
- found top-20 recall,
- evidence rate,
- best severity capture,
- severity regret,
- post-action violation and violation reduction,
- action cost,
- invalid tool calls,
- schema repairs and type coercions,
- duplicate validation requests,
- explicit submission and auto-finalization indicators,
- validation budget use.

These metrics distinguish answer quality, tool evidence, search quality, mitigation quality, and workflow compliance.

## Development Notes

- Use Level 1 to test basic steady-state action submission and physical validation.
- Use Level 2 to test agentic behavior, tool use, validation-budget allocation, and LLM workflows.
- Keep hidden oracle quantities and private endpoint URLs outside the public repository.
- Regenerate results after modifying prompts, adapters, scoring rules, or case-generation settings.
