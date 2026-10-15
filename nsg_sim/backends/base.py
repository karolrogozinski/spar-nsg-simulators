"""Model backends: one async `complete` call per prompt, registered by name.

Add a provider by subclassing `Backend` and decorating it with `@register("name")`; experiments
then select it with `backend: name`. OpenAI-compatible providers only need a preset in
openai_compat.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from nsg_sim.config import RunSettings

REGISTRY: dict[str, type["Backend"]] = {}


def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


@dataclass
class Request:
    prompt: str
    record: dict | None = None        # baselines read labels from the record; API backends ignore it
    with_explanation: bool = True


@dataclass
class Response:
    text: str | None
    reasoning: str | None = None
    finish_reason: str | None = None
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    provider: str | None = None       # who served the request, when the API says (e.g. OpenRouter)


class Backend:
    name: str = "base"
    calls_api: bool = True            # False for in-process baselines (no cost, no request limit)

    def __init__(self, sim, run_settings: RunSettings | None = None):   # sim: config.Simulator
        self.sim = sim
        self.run = run_settings or RunSettings()

    async def complete(self, request: Request) -> Response:
        raise NotImplementedError

    async def close(self):
        pass


def make_backend(sim, run_settings=None) -> Backend:
    # import providers so they register themselves
    from nsg_sim.backends import baselines, openai_compat  # noqa: F401
    if sim.backend not in REGISTRY:
        raise ValueError(f"unknown backend {sim.backend!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[sim.backend](sim, run_settings)
