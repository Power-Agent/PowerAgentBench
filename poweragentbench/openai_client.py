"""OpenAI Responses API client used by the PowerAgentBench-SS LLM adapter.

This client intentionally keeps deployment details outside the code. API keys,
model names, reasoning settings, timeouts, and output limits can be supplied by
command-line flags or environment variables loaded by the runner script.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

Message = Dict[str, str]

TOOL_COMMAND_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "description": "One public benchmark tool name.",
        },
        "args": {
            "type": "object",
            "description": "Arguments for the selected tool.",
            "additionalProperties": True,
        },
    },
    "required": ["tool", "args"],
    "additionalProperties": False,
}

TRANSIENT_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAIResponsesClient:
    """Small dependency-free callable client for the OpenAI Responses API.

    The class follows the same callable interface as OllamaGenerateClient:
        client(messages: list[dict[str, str]]) -> str

    It returns the model's visible text output only. Debug logs should use
    sanitized_debug() so API keys, raw outputs, and model-internal reasoning are
    not written to benchmark artifacts.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.5",
        url: str = "https://api.openai.com/v1/responses",
        temperature: float | None = None,
        max_output_tokens: int | None = 4096,
        structured_outputs: bool = True,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.api_key = api_key
        self.model = model
        self.url = url
        self.temperature = None if temperature is None else float(temperature)
        self.max_output_tokens = None if max_output_tokens is None else int(max_output_tokens)
        self.structured_outputs = bool(structured_outputs)
        self.reasoning_effort = reasoning_effort or None
        self.reasoning_summary = reasoning_summary or None
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.last_debug: Dict[str, Any] | None = None
        self.last_client_warning: str | None = None
        self.last_error: str | None = None
        self.retry_count_last_call: int = 0

    def __call__(self, messages: List[Message]) -> str:
        payload = self._payload(messages)
        out = self._post(payload)
        return self._extract_text(out)

    def _payload(self, messages: List[Message]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": m["role"],
                    "content": m["content"],
                }
                for m in messages
            ],
            "store": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        if self.structured_outputs:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "poweragentbench_tool_command",
                    "description": "A single PowerAgentBench public tool command.",
                    "schema": TOOL_COMMAND_SCHEMA,
                    "strict": False,
                }
            }
        else:
            payload["text"] = {"format": {"type": "json_object"}}
        reasoning: Dict[str, Any] = {}
        if self.reasoning_effort:
            reasoning["effort"] = self.reasoning_effort
        if self.reasoning_summary:
            reasoning["summary"] = self.reasoning_summary
        if reasoning:
            payload["reasoning"] = reasoning
        return payload

    def _post(self, payload: Dict[str, Any], *, allow_temperature_retry: bool = True) -> Dict[str, Any]:
        self.retry_count_last_call = 0
        self.last_error = None
        last_exc: BaseException | None = None

        for attempt in range(self.max_retries + 1):
            try:
                out = self._post_once(payload)
                self.retry_count_last_call = attempt
                self.last_debug = out
                return out
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_exc = exc
                self.last_error = f"HTTP {exc.code}: {body[:500]}"
                if (
                    allow_temperature_retry
                    and exc.code == 400
                    and "temperature" in body.lower()
                    and payload.get("temperature") is not None
                ):
                    # Some reasoning models reject temperature entirely. If a user
                    # supplied it through an old .env file, retry once without it.
                    retry_payload = dict(payload)
                    retry_payload.pop("temperature", None)
                    self.temperature = None
                    self.last_client_warning = (
                        "OpenAI model rejected temperature. Retried once with temperature omitted."
                    )
                    return self._post(retry_payload, allow_temperature_retry=False)
                if exc.code not in TRANSIENT_HTTP_STATUS or attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI API request failed with HTTP {exc.code}: {body}") from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                last_exc = exc
                self.last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OpenAI API request failed after {attempt + 1} attempt(s): {self.last_error}"
                    ) from exc

            self._sleep_before_retry(attempt)

        raise RuntimeError(f"OpenAI API request failed: {last_exc}") from last_exc

    def _post_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff <= 0:
            return
        delay = self.retry_backoff * (2 ** attempt)
        time.sleep(delay)

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]

        chunks: List[str] = []
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for content in item.get("content", []) or []:
                    if isinstance(content, dict) and content.get("type") == "output_text":
                        text = content.get("text")
                        if isinstance(text, str):
                            chunks.append(text)
            elif item.get("type") == "output_text":
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()

    def sanitized_debug(self) -> Dict[str, Any]:
        """Return API diagnostics without raw output text or reasoning content."""
        raw = self.last_debug or {}
        out: Dict[str, Any] = {}
        for key in (
            "id",
            "model",
            "created_at",
            "status",
            "error",
            "incomplete_details",
        ):
            if key in raw:
                out[key] = raw[key]
        usage = raw.get("usage")
        if isinstance(usage, dict):
            out["usage"] = usage
        out["output_text_chars"] = len(self._extract_text(raw))
        out["structured_outputs"] = self.structured_outputs
        out["temperature"] = self.temperature
        out["max_output_tokens"] = self.max_output_tokens
        out["reasoning_effort"] = self.reasoning_effort
        out["reasoning_summary"] = self.reasoning_summary
        out["timeout"] = self.timeout
        out["max_retries"] = self.max_retries
        out["retry_backoff"] = self.retry_backoff
        out["retry_count_last_call"] = self.retry_count_last_call
        if self.last_client_warning:
            out["client_warning"] = self.last_client_warning
        if self.last_error:
            out["last_error"] = self.last_error
        return out
