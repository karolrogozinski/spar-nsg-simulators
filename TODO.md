# TODO

## Next experiments: which simulators make NSG work?

**Question.** Does NSG separate faithful from unfaithful explanations, and how does that depend
on the simulator (size, reasoning, model family) and on the explanation's specificity (P0–P3)?

**Main quantity.** The gap NSG(faithful) − NSG(unfaithful), per level and simulator. Always
report it next to `acc` and `acc_wo`, since NSG is unstable when `acc_wo` is high. NSG on faithful
explanations alone cannot tell a useful simulator from one that follows any explanation blindly.

### Dataset
- [ ] New recipe `configs/datasets/triage_models_v1.yaml`:
  - 12–16 sampled rules (`preset: medium`, some `hard`, so complexity can be studied too),
  - `faithful` + `omission` only; omission is the cleanest contrast, while substitution proxies
    are chosen automatically and add noise,
  - levels P0–P3, no evidence,
  - `counterfactuals: {n_reference: 4, n_per_reference: 8}` for enough reference patients per cell.
- [ ] Mirror twin `triage_models_counter_v1` (`derive_from` + `transform: mirror`) for the prior-vs-explanation question (H2).
- [ ] `nsg preview` the texts and run the baseline smoke test before any model.

### Experiments, in this order
- [ ] **E1 Reasoning (pilot).** `openai/gpt-oss-20b` at `reasoning_effort` low / medium / high:
  same weights, one setting changed, the cleanest contrast and the cheapest. Optionally Qwen3-32B
  with thinking on / off. Raise `run.max_tokens` for `high`.
- [ ] **E2 Size.** Gemma 3 4B / 12B / 27B (no reasoning), plus a second family to check that the
  trend holds (Qwen3 14B / 32B, or Llama 3.1 8B / 70B).
- [ ] **E3 Family.** About 20–30B models from different labs: gemma-3-27b, qwen3-32b,
  mistral-small-3.2-24b, gpt-oss-20b. Descriptive: family changes everything at once.
- [ ] Before each full run: a short run of ~20 cases per model, then check the `invalid` column
  (small models may not keep the answer format), then `--dry-run` for the request count and cost.

### Hypotheses to write down before running
- **H1.** The P0 → P3 slope is steeper for stronger simulators: they can use exact thresholds,
  weak ones guess anyway.
- **H2.** Stronger simulators rely more on their priors: on the mirror (counter-intuitive) twin
  they follow faithful explanations less, which would bias NSG against counter-intuitive but
  faithful explanations.
- **H3.** Reasoning helps mostly on complex rules (more conditions, nesting), not on simple ones.

### Later
- [ ] Substitution and wrong_threshold; evidence conditions (k0–k5).
- [ ] LLM-written levels (P2–P4 from a description of the level) as an alternative to template wording.

## Pipeline

- [ ] Simulator `tags` (e.g. `{family: gemma3, size_b: 12}`) carried into results as
      `model.family` / `model.size_b`, usable in `--by`, `--where`, and as a numeric x-axis in `nsg plot`.
- [ ] Live test of the Ollama backend (second laptop) with `nsg ping ollama gpt-oss:20b-cloud`.
- [ ] Live test of LLM-written levels on a few rules, with a manual review of the texts.
- [ ] Live test of `nsg push` / `nsg pull` (needs `HF_TOKEN`; push creates a repo in the org).
- [ ] Option to make substitution proxies keep the intuitive direction, or pin them in recipes
      used for teaching (`triage_simple_v1` currently gets e.g. "oxygen ≥ 92%" for "temperature ≥ 39").
- [ ] README: the clone URL in Quick Start points to another repository name.

## Done (2026-10-13)

- `nsg_sim` package:
  - `nsg` command: `validate` / `preview` / `build` / `run` / `analyze` / `plot` / `report` / `ping` / `push` / `pull`,
  - YAML domains, dataset recipes and experiments,
  - 57 tests.
- Step-by-step guide: `docs/ADDING_A_DATASET.md`.
- Examples on one domain (`triage`):
  - `triage_simple_v1`: 4 rules, defaults,
  - `triage_full_v1`: every option.
- OpenRouter connection tested: gpt-oss-20b, 80/80 answers parsed.
