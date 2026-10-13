"""Tabular domains defined entirely in YAML (configs/domains/<name>.yaml).

Profile templates use {feature} placeholders. A feature's `display` map turns a value into
words (e.g. has_default: {true: "a", false: "no"}), and {feature|s} adds a plural "s" unless the
value is 1. Without a `profile`, inputs are described generically ("feature = value, ...").
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import yaml

from nsg_sim.domain.base import Domain
from nsg_sim.domain.features import feature_from_config

DOMAINS_DIR = Path(__file__).resolve().parents[2] / "configs" / "domains"
PLACEHOLDER = re.compile(r"\{(\w+)(?:\|(\w+))?\}")


class TabularDomain(Domain):
    def __init__(self, name, features, task=None, explanation=None, profile: str | None = None,
                 display: dict | None = None):
        super().__init__(name, features, task, explanation)
        self.profile = profile
        self.display = display or {}

    def describe(self, x: dict) -> str:
        if not self.profile:
            return self.generic_description(x)

        def fill(match):
            name, modifier = match.group(1), match.group(2)
            value = x[name]
            if modifier == "s":
                return "" if value == 1 else "s"
            shown = self.display.get(name, {})
            key = str(value).lower() if isinstance(value, bool) else value
            return str(shown.get(key, value))

        return PLACEHOLDER.sub(fill, self.profile)


def load_domain(ref: str | Path) -> Domain:
    """Load a domain by name (configs/domains/<name>.yaml) or path.

    Names are looked up in configs/domains/ and then in the folders listed in NSG_DOMAIN_PATH
    (separated by ":"), e.g. domains kept outside the repo or test fixtures.
    A YAML file with `class: package.module:ClassName` loads a custom Domain subclass,
    which receives the parsed YAML as keyword arguments.
    """
    path = Path(ref)
    if not path.suffix:
        path = domain_path(str(ref))
    cfg = yaml.safe_load(path.read_text())
    if "class" in cfg:
        module, cls = cfg.pop("class").split(":")
        return getattr(importlib.import_module(module), cls)(**cfg)
    return domain_from_config(cfg)


def domain_path(name: str) -> Path:
    dirs = [DOMAINS_DIR] + [Path(d) for d in os.environ.get("NSG_DOMAIN_PATH", "").split(os.pathsep) if d]
    for d in dirs:
        if (d / f"{name}.yaml").exists():
            return d / f"{name}.yaml"
    raise FileNotFoundError(f"domain {name!r} not found in {', '.join(str(d) for d in dirs)}")


ALLOWED_KEYS = {"name", "description", "task", "explanation", "features", "profile"}


def domain_from_config(cfg: dict) -> TabularDomain:
    unknown = set(cfg) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"domain {cfg.get('name')}: unknown keys {sorted(unknown)}")
    features, display = {}, {}
    for name, fcfg in cfg["features"].items():
        features[name] = feature_from_config(name, fcfg)
        if "display" in fcfg:
            display[name] = {str(k).lower() if isinstance(k, bool) else k: v for k, v in fcfg["display"].items()}
    return TabularDomain(cfg["name"], features, cfg.get("task"), cfg.get("explanation"),
                         cfg.get("profile"), display)
