# Level 2: Agentic N-2 Steady-State Evaluation

This task evaluates whether scripted or LLM agents can search N-2 contingencies, spend a limited validation budget, submit evidence-backed rankings, and optionally mitigate hidden top-20 violations.

The default case source is the existing IEEE 39-bus case distributed in this repository. The runner converts it to a lightweight DC representation and creates deterministic operating-point variants from fixed seeds. A synthetic fallback is also available for development.

## Goal

For each case, an agent should:

1. inspect the public case summary,
2. use public screening tools to prioritize N-2 candidates,
3. spend the limited validation budget on promising candidates,
4. submit a ranked list supported by validation evidence,
5. optionally call redispatch to reduce hidden post-action violations.

## Public and Hidden Split

The agent sees the public case summary, candidate count, tool API, validation budget, prompt template, and tool observations. The evaluator separately computes the hidden oracle over all N-2 candidates and returns discovery, evidence, mitigation, efficiency, and reliability metrics.

The agent never sees hidden oracle labels during the run.

## Public Tools

The Level 2 LLM interface exposes the following tools:

- `case_summary`: return network size, base severity, candidate count, and remaining budget.
- `rank_base_loading`: rank candidates by public base-flow stress.
- `rank_lodf`: rank candidates using an LODF-style approximate screen.
- `validate`: run exact public validation for selected candidates and consume budget.
- `redispatch`: apply bounded preventive redispatch on a focus set.
- `submit`: submit ranked contingencies, mitigation, and diagnosis.

## Run Scripted Baselines

```bash
python scripts/run_steady_n2_baselines.py \
  --case-source case39 \
  --cases 8 \
  --budget 80 \
  --report-k 20
```

Outputs are written to:

```text
results/steady_n2/
```

The baseline runner writes per-case CSVs, aggregate CSVs, and LaTeX table rows.

## Configure Ollama Without Committing Private Endpoints

Copy the example environment file and edit it locally:

```bash
cp benchmarks/steady/level_2/.env.example benchmarks/steady/level_2/.env
```

The local `.env` file is ignored by Git. Use it for private or internal network endpoints such as a non-public Ollama server.

Required setting:

```bash
POWERAGENTBENCH_OLLAMA_URL=http://localhost:11434/api/generate
```

Recommended model list:

```bash
POWERAGENTBENCH_OLLAMA_MODELS=qwen3.5:latest mistral-nemo:12b command-r:35b
```

Recommended generation settings used in the paper experiments:

```bash
POWERAGENTBENCH_OLLAMA_TEMPERATURE=0.0
POWERAGENTBENCH_OLLAMA_NUM_CTX=16384
POWERAGENTBENCH_OLLAMA_API_MODE=generate
POWERAGENTBENCH_OLLAMA_THINK=false
POWERAGENTBENCH_OLLAMA_SCHEMA_FORMAT=true
```

The corresponding Ollama request includes:

```json
"options": {
  "temperature": 0.0,
  "num_ctx": 16384
}
```

## Run Deployed Ollama Agents

After configuring `.env`, run:

```bash
python scripts/run_steady_n2_ollama_eval.py \
  --case-source case39 \
  --cases 8 \
  --budget 80 \
  --report-k 20 \
  --max-turns 12 \
  --prompt-template benchmarks/steady/level_2/prompts/steady_n2_llm_prompt.json
```

You can override the model list from the command line:

```bash
python scripts/run_steady_n2_ollama_eval.py \
  --models qwen3.5:latest mistral-nemo:12b gpt-oss:20b command-r:35b \
  --case-source case39 \
  --cases 8 \
  --budget 80 \
  --report-k 20 \
  --max-turns 12 \
  --prompt-template benchmarks/steady/level_2/prompts/steady_n2_llm_prompt.json
```

The runner writes per-case CSVs, aggregate CSVs, tool logs, API debug files, and LaTeX table rows under:

```text
results/steady_n2/
```

## Prompt Template

The default prompt template is:

```text
benchmarks/steady/level_2/prompts/steady_n2_llm_prompt.json
```

The prompt specifies the JSON command schema, allowed tools, validation budget, canonical contingency representation, and expected workflow. Keeping the prompt template in the repository makes multi-model LLM comparisons reproducible.

## Output Metrics

The runner returns per-case and aggregate CSV files with:

- `validated_calls`,
- `reported_top20_recall`,
- `validated_top20_recall`,
- `found_top20_recall`,
- `evidence_rate`,
- `best_capture_validated`,
- `severity_regret`,
- `post_top20_violation`,
- `violation_reduction`,
- `action_cost`,
- `invalid_tool_calls`,
- `schema_repairs`,
- `type_coercions`,
- `duplicate_validation_requests`,
- `submitted_explicitly`,
- `auto_finalized`,
- `validation_budget_used`,
- `anytime_risk_auc`,
- `anytime_risk_at_25`,
- `anytime_risk_at_50`,
- `anytime_risk_at_75`,
- `anytime_risk_at_100`.

These fields separate search quality, evidence quality, tool compliance, budget use, mitigation, and workflow completion.

### Severity-Weighted Anytime Risk Discovery

The evaluator reports severity-weighted anytime risk discovery for Level 2 through:

- `anytime_risk_auc`,
- `anytime_risk_at_25`,
- `anytime_risk_at_50`,
- `anytime_risk_at_75`,
- `anytime_risk_at_100`.

Dangerous contingencies are selected using the explicit severity threshold when one is supplied; otherwise the evaluator uses the configured empirical severity quantile (`danger_quantile`, default `0.95`). Values at or above the cutoff are included.

Discovery is based on the first completed unique pre-action validation evaluations in their recorded order. For batched validation, candidate order within each batch matters. Duplicate requests do not create additional trajectory positions, and post-action validations do not contribute to this trajectory. A validated contingency contributes here whether or not it appears in `reported`; this is distinct from the evidence-backed submission metrics.

Each discovered dangerous contingency contributes its non-negative hidden oracle severity. The risk-discovery fraction is the accumulated discovered severity divided by the total non-negative hidden severity of the dangerous set. Hidden oracle values are used only by the evaluator and are never exposed to the agent.

`anytime_risk_auc` is the normalized trapezoidal area under the discovery curve, normalized by the allocated validation budget. The horizontal axis measures validation-count progress, not elapsed time or LLM turns. Percentage checkpoints are evaluated at the exact fractional budget positions on the linearly interpolated curve; they are not rounded counts of completed validations. Thus, for example, a 75% checkpoint at budget 10 is evaluated at position 7.5.

If the agent consumes fewer validations than the allocated budget, its final discovery level is held constant over the unused budget. A non-positive budget or total dangerous severity at or below `1e-12` produces zero-valued metrics.

The per-case metrics complement final discovery, evidence, safety, mitigation, and action-cost metrics rather than replacing them. In particular, `anytime_risk_at_100` is not generally the complement of `severity_weighted_false_negative`, because anytime discovery is based on validated contingencies while the evidence-backed metric requires reported contingencies supported by validation.

For aggregate outputs, distinguish the per-case metric names above from their corresponding `_mean` and `_std` columns. The metric catalogue is descriptive rather than an exhaustive listing of every field returned by `score_agent()`.

## Evaluation Regimes

- **Open**: users can inspect public files and debug agents locally.
- **Sealed**: agents interact only with the public tool server while oracle labels and evaluator scripts remain private.
- **Stress**: agents are rerun across seeds, prompt variants, or scenario variants to measure reliability.

## Notes

- The Level 2 evaluator is a lightweight DC approximation intended to test the benchmark mechanics.
- It is not a replacement for AC security analysis.
- The same public-agent and hidden-evaluator protocol can be connected to AC power flow, SCOPF, voltage-security studies, or commercial simulators.
