"""Provider-agnostic LLM JSON-command adapter for PowerAgentBench-SS."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from poweragentbench.steady_state_agentic import (
    AgentOutput,
    Contingency,
    GridCase,
    SteadyN2ToolServer,
)

Message = Dict[str, str]
LLMCallable = Callable[[List[Message]], str]

DEFAULT_SYSTEM_PROMPT = """You are an engineering agent for PowerAgentBench-SS.
Your task is to find high-severity N-2 branch-outage contingencies under a limited validation budget, support submitted claims with tool evidence, optionally mitigate, and submit a ranked list.

Return exactly one JSON object per turn and no prose. Use this schema:
{"tool":"<tool_name>","args":{...}}

Allowed tools and canonical arguments:
1. {"tool":"case_summary","args":{}}
2. {"tool":"rank_base_loading","args":{"top_n":80}}
3. {"tool":"rank_lodf","args":{"top_n":80}}
4. {"tool":"validate","args":{"contingencies":[[2,11],[11,37],[11,43],[11,21],[11,41],[11,45],[11,26],[11,15],[11,16],[11,29]]}}
5. {"tool":"redispatch","args":{"focus":[[2,11],[11,37],[11,43],[11,21],[11,41]]}}
6. {"tool":"submit","args":{"reported":[[2,11],[11,37],[11,43],[11,21],[11,41]],"diagnosis":"brief evidence-backed summary"}}

Important rules:
- A contingency is a pair of branch ids [i,j], not bus ids.
- Branch ids are integers from 0 to n_branch-1.
- Use the canonical field name "contingencies" for validate and "reported" for submit.
- Use ranking tools first, then validate large batches of promising pairs up to the remaining validation budget.
- Do not validate only two example pairs unless the budget is nearly exhausted.
- Do not repeat already validated pairs.
- Call submit before the final turn.
- Do not use case_name, bus, branch/bus dictionaries, natural language, or markdown.
""".strip()

def _strip_wrappers(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _normalize_command(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "tool" in obj:
        obj.setdefault("args", {})
        return obj
    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        return {"tool": fn.get("name", ""), "args": fn.get("arguments", {}) or {}}
    if "name" in obj and "arguments" in obj:
        return {"tool": obj.get("name", ""), "args": obj.get("arguments", {}) or {}}
    if "tool_call" in obj and isinstance(obj["tool_call"], dict):
        tc = obj["tool_call"]
        return {"tool": tc.get("tool", tc.get("name", "")), "args": tc.get("args", tc.get("arguments", {})) or {}}
    raise ValueError("JSON object did not contain a tool command")


def parse_json_command(text: str) -> Dict[str, Any]:
    cleaned = _strip_wrappers(text)
    if not cleaned:
        raise ValueError("empty model response")
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        js = _first_json_object(cleaned)
        if js is None:
            raise ValueError(f"model response did not contain JSON: {cleaned[:300]}")
        obj = json.loads(js)
    if not isinstance(obj, dict):
        raise ValueError("tool command must be a JSON object")
    return _normalize_command(obj)


def load_prompt_template(path: str | Path | None) -> str | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return str(payload.get("system", "")).strip() or None
    return str(payload).strip() or None


class LLMToolAgent:
    """Adapter that turns an LLM callable into a benchmark agent."""

    def __init__(
        self,
        llm: LLMCallable,
        validation_budget: int = 80,
        report_k: int = 20,
        max_turns: int = 12,
        max_rank_return: int = 120,
        name: str = "LLM-agent",
        system_prompt: str | None = None,
        require_submit: bool = False,
    ) -> None:
        self.llm = llm
        self.validation_budget = int(validation_budget)
        self.report_k = int(report_k)
        self.max_turns = int(max_turns)
        self.max_rank_return = int(max_rank_return)
        self.name = name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.require_submit = bool(require_submit)
        self.last_tool_log: List[Dict[str, Any]] = []
        self.last_raw_responses: List[str] = []

    def run(self, case: GridCase, candidates: Sequence[Contingency]) -> AgentOutput:
        server = SteadyN2ToolServer(
            case=case,
            candidates=candidates,
            validation_budget=self.validation_budget,
            report_k=self.report_k,
            max_rank_return=self.max_rank_return,
        )
        messages = self._initial_messages(case, candidates)
        raw_responses: List[str] = []
        final_reported: List[Contingency] | None = None
        invalid = 0
        schema_repairs = 0
        type_coercions = 0
        duplicate_validation_requests = 0
        submitted_explicitly = 0.0
        auto_finalized = 0.0

        for _ in range(self.max_turns):
            response = self.llm(messages) or ""
            raw_responses.append(response)
            messages.append({"role": "assistant", "content": response})
            try:
                command = parse_json_command(response)
                tool = str(command.get("tool", "")).strip()
                args = command.get("args", {}) or {}
                observation, done, reported = server.execute(tool, args)
                if "error" in observation:
                    invalid += 1
                schema_repairs += int(observation.get("schema_repairs", 0) or 0)
                type_coercions += int(observation.get("type_coercions", 0) or 0)
                duplicate_validation_requests += int(observation.get("duplicate_validation_requests", 0) or 0)
                server.state.tool_log.append({"tool": tool, "args": args, "observation": observation})
                messages.append({"role": "user", "content": json.dumps({"observation": observation})})
                if done:
                    final_reported = reported
                    submitted_explicitly = 1.0
                    break
            except Exception as exc:
                invalid += 1
                observation = {
                    "error": str(exc),
                    "instruction": "Return exactly one JSON command with fields 'tool' and 'args'.",
                    "raw_response_prefix": response[:1000],
                }
                server.state.tool_log.append({"tool": "parse_error", "observation": observation})
                messages.append({"role": "user", "content": json.dumps({"observation": observation})})

        if final_reported is None:
            if self.require_submit:
                final_reported = []
            else:
                final_reported = sorted(
                    server.state.validated,
                    key=lambda c: server.state.validated[c],
                    reverse=True,
                )[: self.report_k]
                auto_finalized = 1.0

        output = AgentOutput(
            name=self.name,
            validated=server.state.validated,
            reported=final_reported[: self.report_k],
            mitigated_case=server.state.mitigated_case,
            action_cost=server.state.action_cost,
            tool_log=server.state.tool_log,
            invalid_tool_calls=float(invalid),
            raw_responses=raw_responses,
            schema_repairs=float(schema_repairs),
            type_coercions=float(type_coercions),
            duplicate_validation_requests=float(duplicate_validation_requests),
            submitted_explicitly=float(submitted_explicitly),
            auto_finalized=float(auto_finalized),
            validation_budget=float(self.validation_budget),
        )
        self.last_tool_log = server.state.tool_log
        self.last_raw_responses = raw_responses
        return output

    def _initial_messages(self, case: GridCase, candidates: Sequence[Contingency]) -> List[Message]:
        user = {
            "task": "steady-state N-2 contingency discovery and mitigation",
            "case": {"name": case.name, "n_bus": case.n_bus, "n_branch": case.n_line, "n_gen": case.n_gen},
            "candidate_count": len(candidates),
            "validation_budget": self.validation_budget,
            "report_k": self.report_k,
            "allowed_tools": ["case_summary", "rank_base_loading", "rank_lodf", "validate", "redispatch", "submit"],
            "canonical_json_examples": [
                {"tool": "case_summary", "args": {}},
                {"tool": "rank_lodf", "args": {"top_n": 80}},
                {"tool": "validate", "args": {"contingencies": [[2, 11], [11, 37], [11, 43], [11, 21], [11, 41], [11, 45], [11, 26], [11, 15], [11, 16], [11, 29]]}},
                {"tool": "submit", "args": {"reported": [[2, 11], [11, 37], [11, 43], [11, 21], [11, 41]], "diagnosis": "brief evidence-backed summary"}},
            ],
        }
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": json.dumps(user)}]
