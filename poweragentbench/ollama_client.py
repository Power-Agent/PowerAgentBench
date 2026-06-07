"""Ollama client used by the PowerAgentBench-SS LLM adapter.

The client intentionally keeps deployment details outside the code. Endpoint URLs,
model names, temperature, and context length can be supplied by command-line flags
or environment variables loaded by the runner script.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List

Message = Dict[str, str]

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}


class OllamaGenerateClient:
    def __init__(
        self,
        url: str,
        model: str,
        temperature: float = 0.0,
        num_ctx: int | None = 16384,
        api_mode: str = "generate",
        schema_format: bool = True,
        think: bool | None = False,
        timeout: float = 120.0,
    ) -> None:
        self.url = url
        self.model = model
        self.temperature = float(temperature)
        self.num_ctx = None if num_ctx is None else int(num_ctx)
        self.api_mode = api_mode
        self.schema_format = bool(schema_format)
        self.think = think
        self.timeout = float(timeout)
        self.last_debug: Dict[str, Any] | None = None

    def __call__(self, messages: List[Message]) -> str:
        if self.api_mode == "chat":
            return self._chat(messages)
        return self._generate(messages)

    def _post(self, payload: Dict[str, Any], url: str | None = None) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url or self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        self.last_debug = out
        return out

    def _format_value(self) -> Any:
        return TOOL_SCHEMA if self.schema_format else "json"

    def _options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {"temperature": self.temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return options

    def _generate(self, messages: List[Message]) -> str:
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(),
            "format": self._format_value(),
        }
        if self.think is not None:
            payload["think"] = self.think
        out = self._post(payload)
        return str(out.get("response", ""))

    def _chat(self, messages: List[Message]) -> str:
        url = self.url.replace("/api/generate", "/api/chat")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options(),
            "format": self._format_value(),
        }
        if self.think is not None:
            payload["think"] = self.think
        out = self._post(payload, url=url)
        msg = out.get("message", {}) or {}
        return str(msg.get("content", ""))
