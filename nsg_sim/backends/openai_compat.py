"""OpenAI-compatible chat APIs: Ollama (local and cloud models), OpenRouter, OpenAI, vLLM.

Each preset gives a default base URL, the environment variable holding the key, and how a
`reasoning_effort` setting is sent. Base URLs can be overridden per simulator (`base_url`) or by
environment variable (OLLAMA_BASE_URL, VLLM_BASE_URL, ...).
"""

from __future__ import annotations

import asyncio
import os
import time

from nsg_sim.backends.base import Backend, Request, Response, register

PRESETS = {
    "ollama": dict(base_url="http://localhost:11434/v1", url_env="OLLAMA_BASE_URL", key_env=None,
                   reasoning="reasoning_effort"),
    # OpenRouter: model ids look like "openai/gpt-oss-20b"; reasoning is a nested object.
    # Check https://openrouter.ai/docs before relying on the exact parameter shape.
    "openrouter": dict(base_url="https://openrouter.ai/api/v1", url_env="OPENROUTER_BASE_URL", key_env="OPENROUTER_API_KEY",
                       reasoning="openrouter"),
    "openai": dict(base_url=None, url_env="OPENAI_BASE_URL", key_env="OPENAI_API_KEY", reasoning="reasoning_effort"),
    "vllm": dict(base_url="http://localhost:8000/v1", url_env="VLLM_BASE_URL", key_env="VLLM_API_KEY",
                 reasoning="reasoning_effort"),
}


class OpenAICompatible(Backend):
    preset: str = "openai"

    def __init__(self, sim, run_settings=None):
        super().__init__(sim, run_settings)
        from openai import AsyncOpenAI
        cfg = PRESETS[self.preset]
        base_url = sim.base_url or os.environ.get(cfg["url_env"]) or cfg["base_url"]
        key = os.environ.get(cfg["key_env"]) if cfg["key_env"] else None
        if cfg["key_env"] and not key and self.preset != "vllm":
            raise RuntimeError(f"backend {self.preset}: set {cfg['key_env']} in .env")
        self.client = AsyncOpenAI(base_url=base_url, api_key=key or "not-needed", max_retries=0)
        self.reasoning_style = cfg["reasoning"]

    def _extra(self) -> dict:
        extra = dict(self.sim.params)
        effort = self.sim.reasoning_effort
        if effort:
            if self.reasoning_style == "openrouter":
                extra.setdefault("extra_body", {})["reasoning"] = {"effort": effort}
            else:
                extra["reasoning_effort"] = effort
        return extra

    async def complete(self, request: Request) -> Response:
        max_tokens = self.sim.max_tokens or self.run.max_tokens
        error = None
        for attempt in range(self.run.retries + 1):
            try:
                start = time.monotonic()
                resp = await self.client.chat.completions.create(
                    model=self.sim.model, messages=[{"role": "user", "content": request.prompt}],
                    temperature=self.sim.temperature, max_tokens=max_tokens, timeout=self.run.timeout, **self._extra())
                msg = resp.choices[0].message
                usage = resp.usage
                return Response(
                    text=msg.content,
                    reasoning=getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None),
                    finish_reason=resp.choices[0].finish_reason,
                    latency_s=round(time.monotonic() - start, 2),
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                )
            except Exception as exc:  # network / rate limit / server error: back off, then report
                error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(2 ** attempt)
        return Response(text=None, error=error)

    async def close(self):
        await self.client.close()


for _name in PRESETS:
    register(_name)(type(f"{_name.title()}Backend", (OpenAICompatible,), {"preset": _name}))
