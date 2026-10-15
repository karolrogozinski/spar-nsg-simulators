# From an idea to NSG results, step by step

This guide takes you from nothing to a finished experiment. It uses one running example, a
**triage** dataset: a nurse decides whether a patient is sent to a doctor today. Every file it
mentions exists in the repo, so you can open it next to the guide:

| Step | You write / run | Example in the repo |
| --- | --- | --- |
| 1. Describe the inputs | `configs/domains/<name>.yaml` | `configs/domains/triage.yaml` |
| 2. Describe the dataset | `configs/datasets/<name>_v1.yaml` | `configs/datasets/triage_simple_v1.yaml` |
| 3. Check and build it | `nsg validate`, `nsg preview`, `nsg build` | |
| 4. Sanity-check it for free | a baseline experiment | `configs/experiments/triage_simple_smoke.yaml` |
| 5. Describe the experiment | `configs/experiments/<name>.yaml` | `configs/experiments/triage_simple_openrouter.yaml` |
| 6. Run it | `nsg run` | |
| 7. Look at the results | `nsg analyze`, `nsg plot` | |
| 8. Share the dataset | `nsg push` | |

`triage_simple_v1` is the small example to learn from: four short rules and default settings.
`configs/datasets/triage_full_v1.yaml` uses the same domain and shows every option (nested rules,
pinned edits, all explanation types, evidence), with `configs/experiments/triage_full_gptoss20b.yaml`.

You only write YAML. Python is needed only for inputs that are not a table of values
(see [Going further](#going-further)).

Contents: [Step 0](#step-0--set-up-once) · [Step 1](#step-1--describe-the-inputs-domain) ·
[Step 2](#step-2--describe-the-dataset-recipe) · [Step 3](#step-3--check-preview-build) ·
[Step 4](#step-4--sanity-check-with-baselines) · [Step 5](#step-5--describe-the-experiment) ·
[Step 6](#step-6--run) · [Step 7](#step-7--results) · [Step 8](#step-8--share) ·
[Going further](#going-further) · [When something goes wrong](#when-something-goes-wrong) ·
[Checklist](#checklist)

---

## Step 0 · Set up (once)

```bash
git clone <repo> && cd spar-nsg-simulators
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q                     # ~15 s, no network: everything installed correctly
```

Put endpoints and keys in a file called `.env` in the repo root. It is git-ignored; never put
keys in YAML files or notebooks.

```
OLLAMA_BASE_URL=http://localhost:11434/v1     # or another machine running Ollama
OPENROUTER_API_KEY=sk-or-...
HF_TOKEN=hf_...                               # or run `huggingface-cli login` once
```

---

## Step 1 · Describe the inputs (domain)

A domain says what one case looks like (its features), who decides about it, and how to talk
about it. Create `configs/domains/<name>.yaml`.

### 1.1 The smallest domain that works

```yaml
name: triage
task:
  actor: triage nurse
  actor_short: nurse
  subject: patient
  decision: The nurse decides whether the patient is sent to a doctor today (YES) or not (NO).
features:
  temperature: {type: numeric, range: [35, 41]}
  heart_rate: {type: numeric, range: [40, 160], step: 5}
  chronic: {type: bool}
  arrival: {type: categorical, values: [walk_in, ambulance, referral]}
```

This is already enough to build a dataset. Everything below makes the texts read better and is
optional. Without it you get generic wording such as *"has temperature of at least 38"* and
*"The patient has temperature = 38, ..."*.

### 1.2 `task`: the framing of the prompt

| Key | Used in the prompt as | Triage |
| --- | --- | --- |
| `actor` | "Your task is to study a **triage nurse**'s decision ..." | `triage nurse` |
| `actor_short` | "The **nurse**'s explanation" | `nurse` |
| `subject` | "reference **patient**", "counterfactual **patient**" | `patient` |
| `decision` | one sentence saying what YES and NO mean | `The nurse decides whether ...` |

### 1.3 `features`: three types

```yaml
features:
  temperature:                       # numeric: whole numbers from range[0] to range[1]
    type: numeric
    range: [35, 41]
  heart_rate:
    type: numeric
    range: [40, 160]
    step: 5                          # optional grid step: 40, 45, ..., 160 (default 1)
  chronic:                           # bool: true / false
    type: bool
  arrival:                           # categorical: one of a fixed list
    type: categorical
    values: [walk_in, ambulance, referral]
```

Every feature can also take these keys:

| Key | What it does | Example |
| --- | --- | --- |
| `label` | human name, used at P3 ("driven mainly by the patient's **heart rate**") and in generic wording | `label: heart rate` |
| `phrases` | how a condition on this feature is written (see 1.4) | |
| `overrides` | exact text for one specific condition | `- {op: "<", value: 1, text: "has had symptoms for less than a day"}` |
| `display` | how a bool / category value reads in the case description (see 1.6) | `display: {true: a, false: "no"}` |
| `aliases` | other words that count as naming the feature, for automatic text checks | `aliases: [pulse, bpm]` |
| `sensitive` | marks features like age or sex; `omission` drops these first | `sensitive: true` |
| `yes_when` | what makes a YES more plausible: `high` / `low` (numeric), `true` / `false` (bool), a list of values (categorical). Used by `substitution` (the proxy points the same way) and by the sampler's `direction` | `yes_when: low` (oxygen), `yes_when: [ambulance, referral]` |
| `mirror` | value swaps for categories in the `mirror` (counter-intuitive) transform | `mirror: {walk_in: ambulance, ambulance: walk_in}` |

### 1.4 `phrases`: how conditions are written at each level

Explanations exist at several levels of specificity. The template levels read the wording from
the domain:

| Level | Style | "temperature >= 39" reads |
| --- | --- | --- |
| P0 | `exact` | has a temperature of at least 39°C |
| P1 | `approximate` | has a temperature of roughly 39°C or more |
| P2 | `qualitative` | has a fever |
| P3 | `features` | (only names the feature: "... driven mainly by the patient's body temperature") |

**Numeric features**: one entry per operator, with one text per style; `{v}` is the threshold.

```yaml
temperature:
  type: numeric
  range: [35, 41]
  phrases:
    ">=":
      exact: "has a temperature of at least {v}°C"
      approximate: "has a temperature of roughly {v}°C or more"
      qualitative: "has a fever"
    "<":
      exact: "has a temperature below {v}°C"
      approximate: "has a temperature below roughly {v}°C"
      qualitative: "does not have a high fever"
```

- **Words that depend on the threshold.** At P2 there is no number, so the word must fit the
  threshold: "temperature ≥ 37" is not "a fever". Give a list; the first entry whose `max` is at
  least the threshold is used, the last entry (no `max`) catches the rest:

  ```yaml
  qualitative:
    - {max: 37, text: "does not have a low body temperature"}   # thresholds up to 37
    - {max: 38, text: "has a slightly raised temperature"}      # 38
    - {text: "has a fever"}                                     # 39 and above
  ```

  Write these lists for every numeric feature you will use with random rules, whose thresholds
  can land anywhere in the range.
- **You only need one operator of each pair.** On the grid, `> 37` is the same as `>= 38` and
  `<= 37` the same as `< 38`, so `>` reuses the `>=` texts and `<=` reuses `<`. Write `>=` and
  `<` and every operator is covered.
- **Operators**: `>=`, `>`, `<=`, `<` for numbers; `==`, `!=` for categories and bools.

**Bool features**: one text for each value.

```yaml
chronic:
  type: bool
  phrases: {"true": "has a chronic condition", "false": "has no chronic condition"}
```

**Categorical features**: a text per value, or one template for all values.

```yaml
arrival:
  type: categorical
  values: [walk_in, ambulance, referral]
  phrases:
    values: {walk_in: "arrived on foot", ambulance: "arrived by ambulance", referral: "arrived with a GP referral"}
    not_values: {walk_in: "did not arrive on foot", ambulance: "did not arrive by ambulance", referral: "did not arrive with a GP referral"}

sex:
  type: categorical
  values: [female, male]
  phrases: {"==": "is {v}", "!=": "is not {v}"}     # one template for every value
```

Bools and categories read the same at P0, P1 and P2 (there is no number to hide).

**Use the same words in phrases and in the profile.** If the profile says "arrived on foot", the
explanation must say "arrived on foot" / "did not arrive on foot", not "walked in". A simulator
does not reliably connect different wordings of the same value: gpt-oss-20b misread "did not walk
in" against "arrived on foot" in 4 of its 5 errors on exact explanations (EXP-006).

### 1.5 `explanation`: sentence frames

```yaml
explanation:
  intro: "The patient is sent to a doctor if the patient "        # + conditions
  unless: ", unless the patient "                                   # + exception
  features_only: "The decision was driven mainly by the patient's {features}."   # P3
  distractor: "The patient's {features} were also checked but did not change the decision."
```

All four have generic defaults ("The answer is YES if the case ..."), so set at least `intro`
and `unless`.

### 1.6 `profile`: how a case is shown to the simulator

```yaml
profile: >-
  The patient is a {age}-year-old {sex} who arrived {arrival}, with a temperature of
  {temperature}°C and a heart rate of {heart_rate} bpm. The patient has {chronic} chronic condition.
```

- `{feature}` inserts the value, through the feature's `display` map if it has one:
  `arrival: display: {walk_in: on foot, ambulance: by ambulance, referral: with a GP referral}`,
  `chronic: display: {true: a, false: "no"}` → "has a chronic condition" / "has no chronic condition".
- `{feature|s}` adds a plural "s" unless the value is 1: `symptoms for {symptom_days} day{symptom_days|s}`.
- Without a profile the case is listed generically: "The patient has temperature = 38 and heart
  rate = 95 (temperature: 35-41, heart rate: 40-160); chronic is on."

**Done with Step 1** when every feature has a `label`, numeric features have `>=` and `<`
phrases with threshold lists at `qualitative`, and the profile reads like a sentence. The full
triage domain (`configs/domains/triage.yaml`) shows every key in use.

---

## Step 2 · Describe the dataset (recipe)

A recipe says which rules decide the cases, which explanations the simulator sees, at which
levels, and how many cases there are. Create `configs/datasets/<name>_v1.yaml`. The simple example
is the whole of `configs/datasets/triage_simple_v1.yaml`:

```yaml
name: triage_simple_v1           # folder name in data/; change content -> new name (_v2)
description: 4 short triage rules; faithful, omission and substitution explanations at P0-P3.
domain: triage                   # a file in configs/domains/

rules:
  - id: S1
    rule: ["oxygen", "<", 92]
  - id: S2
    rule: ["temperature", ">=", 39]
  - id: S3
    rule: {all: [["temperature", ">=", 38], ["heart_rate", ">=", 110]]}
  - id: S4
    rule: {any: [["arrival", "==", "ambulance"], ["oxygen", "<", 90]]}

explanations:
  types: [faithful, omission, substitution]
```

Everything left out takes its default: levels P0–P3, 2 reference patients × 8 counterfactuals per
rule and explanation type, no evidence, seed 0. That gives 640 records. All the sections a recipe
can have:

```yaml
name: ...
description: ...
domain: triage
seed: 0                          # same recipe + same seed = identical dataset

rules: [...]                     # 2.1: where the rules come from (pick one source)
explanations: {...}              # 2.2: which explanations
levels: [...]                    # 2.3: which levels (optional: P0-P3 by default)
counterfactuals: {...}           # 2.4: how many cases
evidence: {...}                  # 2.5: optional extra labelled examples
```

### 2.1 Rules: pick one of three sources

#### Option A · Write the rules yourself (`rules`)

Best when the rules should make sense to a person, or when you test specific structures.

A **condition** is `[feature, operator, value]`:

```yaml
["temperature", ">=", 39]          # numeric: >=  >  <=  <
["chronic", "==", true]            # bool: == / != true / false
["arrival", "==", "ambulance"]     # categorical: the value must be in the feature's list
```

**Combine conditions** with groups, nested as deeply as you like (the examples below come from
`triage_full_v1.yaml`):

| You write | Meaning | Reads (P0) |
| --- | --- | --- |
| `["oxygen", "<", 92]` | one condition | has an oxygen saturation below 92% |
| `{all: [A, B]}` | A and B | ... rates their pain at 7/10 or more and has had symptoms for at least 3 days |
| `{any: [A, B]}` | A or B | ... either arrived by ambulance or has an oxygen saturation below 90% |
| `{at_least: 2, of: [A, B, C, D]}` | at least 2 of them | ... meets at least two of these conditions: A; B; C; D |
| `{not: A}` | not A | the opposite condition ("has a temperature below 38°C"), or "meets none of the following" for a group |
| any group + `unless: C` | the group, except when C | ..., unless the patient rates their pain below 2/10 |

```yaml
rules:
  - id: T03
    rule: {all: [["pain", ">=", 7], ["symptom_days", ">=", 3]]}
  - id: T04
    rule: {all: [["temperature", ">=", 38], {any: [["heart_rate", ">=", 110], ["oxygen", "<", 94]]}]}
  - id: T07
    rule: {all: [["age", ">=", 65], ["chronic", "==", true]], unless: ["pain", "<", 2]}
  - id: T08                          # longer rules can be written over several lines
    rule:
      any: [{all: [["surgery", "==", true], ["temperature", ">=", 38]]}, ["oxygen", "<", 90]]
      unless: ["arrival", "==", "referral"]
```

An exception on a single condition is written as a one-item group:
`{all: [["temperature", ">=", 39]], unless: ["age", "<", 18]}`.

Rules that one sentence states clearly stay one sentence. Deeper nesting is written as an indented
list, so it is always clear what each "or" and "unless" covers:

```
The patient is sent to a doctor if the patient meets at least one of the following:
- meets all of the following:
  - had surgery recently
  - has a temperature of at least 38°C
- has an oxygen saturation below 90%
unless the patient arrived with a GP referral.
```

**Optional per rule**: control how the unfaithful explanations change it, and label it.
Anything you leave out is chosen automatically.

```yaml
  - id: T04
    rule: {all: [["temperature", ">=", 38], {any: [["heart_rate", ">=", 110], ["oxygen", "<", 94]]}]}
    omit: ["temperature", ">=", 38]                                    # omission drops this condition
    substitute: [["heart_rate", ">=", 110], ["pain", ">=", 6]]         # substitution: [replaced, proxy]
    # with several edits per explanation (2.2), list them in order:
    #   omit: [["temperature", ">=", 38], ["oxygen", "<", 94]]
    #   substitute: [[["heart_rate", ">=", 110], ["pain", ">=", 6]], [["oxygen", "<", 94], ["symptom_days", ">=", 5]]]
    shift: [["temperature", ">=", 38], ["temperature", ">=", 40]]      # wrong_threshold: [old, new]
    distractors: ["sex", "insurance"]                                  # distractor mentions these
    tags: {level: 3, theme: infection}                                 # your labels, usable in analysis
```

How the automatic choices work:

- **omission** drops conditions, pinned ones first, then sensitive features, then the rest;
- **substitution** swaps a condition for one on an unused numeric feature that a similar share of
  patients meet (a plausible proxy) and that points the same way: if the old condition makes a YES
  more plausible, so does the proxy. The way comes from the features' `yes_when`; without it the
  proxy keeps the old condition's end of the range (`>=` stays `>=`). Categorical conditions have
  no way, so any proxy can replace them;
- **wrong_threshold** moves one numeric threshold by a share of the feature's range (12% by default);
- **distractor** picks two unused features.

How many conditions are changed is set per recipe (2.2). A variant that cannot be built for a rule
is skipped for that rule and listed by `nsg preview` and in the manifest (`skipped_variants`):
there is no omission for a one-condition rule, and substitution needs enough unused numeric features.

#### Option B · Random rules with a difficulty dial (`rule_sampler`)

Best for many rules, or for studying difficulty. Start from a preset and change any setting:

```yaml
rule_sampler:
  preset: medium
  n_rules: 12
```

| Preset | Conditions per rule | Connectors (weights) | Nesting depth | Typical text (P0) |
| --- | --- | --- | --- | --- |
| `easy` | 1–2 | and | 1 | one sentence, ~15 words |
| `medium` | 2–4 | and 3, or 1, unless 1 | 2 | one sentence or a short list |
| `hard` | 4–8 | and 3, or 2, unless 1, at least k of n 1 (k 2–3) | 3 | nested list, ~50 words |
| `very_hard` | 6–10 | and 2, or 2, unless 1, not 1, at least k of n 2 (k 2–4) | 4 | nested list, ~70 words |

All settings (each overrides the preset):

| Setting | What it changes | Default | Example |
| --- | --- | --- | --- |
| `n_rules` | how many rules | 6 | `n_rules: 20` |
| `n_conditions` | `[min, max]` conditions per rule | from preset | `n_conditions: [3, 5]` |
| `connectors` | which connectors appear and how often (weights) | from preset | `connectors: {and: 2, or: 1, at_least: {weight: 1, k: [2, 3]}}` |
| `max_depth` | how deeply groups may nest | from preset | `max_depth: 2` |
| `condition_pass_rate` | `[min, max]` share of patients meeting one condition; narrower = no extreme thresholds | `[0.35, 0.8]` | `condition_pass_rate: [0.4, 0.7]` |
| `rule_yes_rate` | `[min, max]` share of patients the whole rule sends to a doctor | `[0.15, 0.85]` | `rule_yes_rate: [0.3, 0.7]` |
| `features` | which features rules may use | all | `features: {exclude: [sex, insurance]}` or `{include: [temperature, oxygen, pain]}` |
| `ops` | which operators | all six | `ops: [">=", "<", "=="]` |
| `direction` | `intuitive`: every condition points the way the features' `yes_when` says (exceptions the other way); `counter`: the opposite; features without `yes_when` are left out | `any` | `direction: intuitive` |

Connector names: `and`, `or`, `unless`, `at_least` (with `k: [min, max]`), `not`. A weight of 0
or leaving a connector out disables it. Each feature appears at most once in a sampled rule.
Every sampled rule is tagged with `n_conditions`, `depth`, `structure` and `yes_rate` for the
analysis.

```yaml
# example: "medium, but longer and only and/or, no demographic features"
rule_sampler:
  preset: medium
  n_rules: 10
  n_conditions: [3, 5]
  connectors: {and: 1, or: 1}
  features: {exclude: [age, sex, insurance]}
```

#### Option C · Rules of growing size (`rule_builder: chains`)

Best for "same conditions, more of them" comparisons. From an ordered list of conditions it
builds rules from the first 1, 2, 3, ... conditions joined by "and", plus or / unless /
at-least versions of the first four:

```yaml
rule_builder:
  kind: chains
  and_sizes: [1, 2, 3, 4, 6]          # rules with the first 1, 2, 3, 4, 6 conditions
  structures: [or, unless, at_least]  # extra rules over the first `structure_size` conditions
  structure_size: 4
  at_least_k: 2
  chains:
    A:
      conditions: [["temperature", ">=", 38], ["heart_rate", ">=", 110], ["pain", ">=", 7],
                   ["oxygen", "<", 94], ["symptom_days", ">=", 3], ["age", ">=", 65]]
      unless: ["arrival", "==", "referral"]   # the exception used by the `unless` structure
```

### 2.2 `explanations`: what the simulator is told

```yaml
explanations:
  types: [faithful, omission, substitution, wrong_threshold, distractor]
  levels: {wrong_threshold: [P0, P1], distractor: [P0]}    # optional; other types get every level
  n_distractors: 2                                          # optional
```

| Type | The explanation describes | Sound? |
| --- | --- | --- |
| `faithful` | the true rule | yes |
| `omission` | the rule with one condition left out | no |
| `substitution` | the rule with one condition replaced by a plausible proxy | no |
| `wrong_threshold` | the rule with one threshold moved | no |
| `distractor` | the true rule, plus features said to be checked but irrelevant | yes |

`wrong_threshold` only makes sense where thresholds are visible (P0, P1), and `distractor`
needs only one level.

**How unfaithful (optional).** By default each unfaithful explanation changes one thing. To study
how NSG reacts to more unfaithfulness, ask for several sizes; each becomes its own variant:

```yaml
explanations:
  types: [faithful, omission, substitution, wrong_threshold]
  omission: {n: [1, 2, 3]}             # drop 1, 2 or 3 conditions
  substitution: {n: [1, 2]}            # replace 1 or 2 conditions
  wrong_threshold: {shift: [0.1, 0.3]} # move a threshold by 10% or 30% of the feature's range
```

- Sizes are nested: omitting 2 conditions drops the 1-condition omission's condition plus one
  more, and a larger shift moves the same threshold the same way, only further. So a larger size
  is strictly more unfaithful.
- Records get an `edit_size` field, and results are labelled e.g. `omission n=2` or
  `wrong_threshold shift=0.3` (`nsg analyze --by explanation`, or `--by explanation_type,edit_size`).
- A size needs a rule big enough: dropping `n` conditions needs at least `n + 1`, replacing `n`
  needs `n` conditions and `n` unused numeric features. To compare sizes on the same rules, use rules
  of a fixed length, e.g. `rule_sampler: {n_conditions: [4, 4]}`.

### 2.3 `levels`: how specific the explanations are

Leave `levels` out to get P0–P3 from the domain's phrases. To change them, list the levels you
want; each is written either from the domain's phrases (`template`) or by a language model from
your description (`llm`):

```yaml
levels:
  - {id: P0, style: exact}                 # template: exact | approximate | qualitative | features
  - {id: P1, style: approximate}
  - id: P2
    style: qualitative
    checks: {forbid_digits: true}          # stop the build if a text breaks the level
  - id: P4
    renderer: llm
    model: {backend: ollama, model: "gpt-oss:20b-cloud", reasoning_effort: low}
    description: Say only which single feature mattered most for the decision, in one short sentence.
    examples:                              # optional: show the model what you mean
      - rule: "The patient is sent to a doctor if the patient has a temperature of at least 39°C and a heart rate of at least 110 bpm."
        text: "The patient's temperature mattered most."
    checks: {forbid_digits: true, max_words: 25, judge: true}
    retries: 2                             # optional, default 2
```

| Level key | Meaning |
| --- | --- |
| `id` | any name; used in filters and plots |
| `renderer` | `template` (default) or `llm` |
| `style` | template only: `exact`, `approximate`, `qualitative`, `features` |
| `formal` | also store the rule as a formula (default: only for `exact`), for `explanation_format: formal` prompts |
| `model` | llm only: same fields as an experiment simulator (Step 5) |
| `description` | llm only: what texts at this level should look like |
| `examples` | llm only: `[{rule: ..., text: ...}]` |
| `checks` | automatic checks, see below |
| `retries` | llm only: new attempts when a text fails the checks |

| Check | Fails when |
| --- | --- |
| `forbid_digits: true` | the text contains a digit |
| `require_thresholds: true` | a threshold of the rule is missing from the text |
| `require_all_features: true` | a feature of the rule is not named (labels and `aliases` count) |
| `forbid_other_features: true` | a feature outside the rule is named |
| `max_words: 40` | the text is longer |
| `judge: true` | the model, asked separately, says the text does not agree with the rule |

LLM texts are generated once when the dataset is built (temperature 0), cached in
`data/_cache/llm_render/`, and stored in the dataset, so a built dataset never needs the model
again. Read every LLM-written level with `nsg preview` before using it.

### 2.4 `counterfactuals`: how many cases per rule

For every rule the builder picks **reference patients** (the case the explanation is about),
then **counterfactual patients** that differ from the reference in 1–2 features. The simulator
predicts the decision on each counterfactual.

```yaml
counterfactuals: {n_reference: 2, n_per_reference: 8, balance: exact}
```

| Key | Meaning | Default |
| --- | --- | --- |
| `n_reference` | reference patients per rule and explanation type (alternating YES / NO) | 2 |
| `n_per_reference` | counterfactuals per reference | 8 |
| `balance` | `exact`: half YES, half NO for every reference. `loose`: spread over YES/NO and over cases where the explanation is right or wrong, as far as they exist | `loose` |
| `pool_size` | how many candidate perturbations are tried per reference | 3000 |

Records per explanation type and level = rules × `n_reference` × `n_per_reference`.

### 2.5 `evidence` (optional): extra labelled examples in the prompt

```yaml
evidence:
  n: 5                                    # examples shown with the case
  contradicting: [0, 1, 3, 5]             # conditions k0, k1, k3, k5: how many contradict the explanation
  types: [omission, substitution]         # explanation types that get evidence
  other_types_contradicting: [0]          # other types get these conditions (k0 = consistent examples)
  levels: [P0]                            # levels that get evidence
```

Each record gets an `evidence_condition`: `none` (no evidence) or `k0`, `k1`, ... Leave
`evidence` out for no evidence.

### 2.6 Paired variants (optional)

A second recipe can transform a built dataset. It keeps every record and its labels and changes
only feature values, thresholds and texts:

```yaml
name: triage_simple_counter_v1
derive_from: triage_simple_v1
transform: mirror          # mirror: thresholds reflected ("fever" becomes "no raised temperature")
                           # abstract: features renamed A, B, C..., numbers rescaled to 0-100
```

Each record points to its original through `paired_record_id`, and the analysis can group by
`variant`.

---

## Step 3 · Check, preview, build

```bash
nsg validate configs/datasets/triage_simple_v1.yaml     # typos, unknown features, wrong values
nsg preview  configs/datasets/triage_simple_v1.yaml     # read the texts, saves nothing
nsg build    configs/datasets/triage_simple_v1.yaml -v  # save to data/triage_simple_v1/
```

`nsg preview` prints, for the first rules (`--rules 5`, `--rule S3`, `--rule all`), the rule and
its text for every explanation type and level, then one example case:

```
=== S3   temperature >= 38 and heart_rate >= 110
  faithful         P0    The patient is sent to a doctor if the patient has a temperature of at least 38°C and has a heart rate of at least 110 bpm.
  faithful         P2    The patient is sent to a doctor if the patient has a slightly raised temperature and has a fast heart rate.
  omission         P0    The patient is sent to a doctor if the patient has a temperature of at least 38°C.
  substitution     P0    The patient is sent to a doctor if the patient has a temperature of at least 38°C and has an oxygen saturation below 89%.
  ...
=== example record S3|omission|r0|c0|P0|none
  reference      The patient is a 17-year-old female ..., with a temperature of 41°C, a heart rate of 140 bpm, ...  -> YES
  counterfactual The patient is a 17-year-old female ..., with a temperature of 41°C, a heart rate of 60 bpm, ...  -> true NO, explanation says YES  (discriminating)
```

Read it as the simulator would. Typical fixes:

| You see | Fix |
| --- | --- |
| "has temperature of at least 38" (no units, clumsy) | add `phrases` for that feature |
| a P2 word that does not fit the threshold ("has a fever" for ≥ 37) | a threshold list at `qualitative` |
| a sampled rule that makes no sense | `features: {exclude: [...]}`, narrower `condition_pass_rate`, or write rules by hand |
| an explanation type missing for a rule | expected for one-condition rules (nothing to omit); otherwise pin `omit` / `substitute` |

After `nsg build -v`, check the summary:

- **records per cell**: every type × level you expect is there;
- **`yes_share`**: about 0.5 (exactly 0.5 with `balance: exact`);
- **`discriminating_share`**: the share of counterfactuals where following the explanation gives
  the wrong answer. Only these separate a faithful explanation from an unfaithful one. If it is
  low (below ~0.15), raise `n_per_reference` or use `balance: loose`.

A built dataset is **immutable**. Building the same recipe again does nothing. If you changed the
recipe or the domain and the content differs, the build stops: give the recipe a new name (`_v2`).
Old results stay valid this way.

---

## Step 4 · Sanity-check with baselines

Baselines are simple strategies that need no model and cost nothing:

```yaml
# configs/experiments/triage_simple_smoke.yaml
name: triage_simple_smoke
dataset: triage_simple_v1
simulators:
  - {backend: baseline, model: follow_g}          # always does what the explanation says
  - {backend: baseline, model: true_f}            # knows the true rule
  - {backend: baseline, model: copy_reference}    # repeats the reference decision
```

A fourth baseline, `majority_evidence` (majority label of the evidence examples), is useful for
datasets with evidence.

```bash
nsg run configs/experiments/triage_simple_smoke.yaml
nsg analyze runs/triage_simple_smoke --by simulator,explanation_type
```

What a healthy dataset shows (real output for `triage_simple_v1`):

| Baseline | Expect | triage_simple_v1 |
| --- | --- | --- |
| `follow_g`, faithful | NSG = 1 | 1.0 |
| `follow_g`, omission / substitution | NSG clearly below 1 | 0.25 / −0.20 |
| `true_f` | accuracy = 1 | 1.0 |
| `copy_reference` | accuracy ≈ 0.5 | 0.50–0.53 |

If `follow_g` reaches NSG 1 on an unfaithful type, its explanations never mislead and that type
measures nothing. Look at its rules in `nsg preview`.

---

## Step 5 · Describe the experiment

```yaml
# configs/experiments/triage_simple_openrouter.yaml
name: triage_simple_openrouter
dataset: triage_simple_v1
sample: {per_cell: 1, seed: 0}
simulators:
  - {backend: openrouter, model: "openai/gpt-oss-20b", reasoning_effort: low}
run: {concurrency: 8, max_requests: 100}
```

Only `name`, `dataset` and `simulators` are required. A fuller setup with every section is
`configs/experiments/triage_full_gptoss20b.yaml`:

```yaml
name: triage_full_gptoss20b
dataset: triage_full_v1
filter: {evidence_conditions: [none]}
sample: {per_cell: 2, seed: 0}
prompt: {template: nsg_paper, instruction: follow, explanation_format: text}
with_baseline: true
simulators:
  - {backend: ollama, model: "gpt-oss:20b-cloud", reasoning_effort: low}
  - {backend: baseline, model: follow_g}
run: {concurrency: 4, max_tokens: 2048, max_requests: 1500}
```

| Key | Meaning | Example |
| --- | --- | --- |
| `name` | results go to `runs/<name>/`; a new setup gets a new name | `triage_simple_openrouter` |
| `description` | free text | |
| `dataset` | a built dataset | `triage_simple_v1` |
| `dataset_sha` | optional: refuse to run unless the dataset hash starts with this | `760f8c2b` |
| `filter.explanation_types` | only these types | `[faithful, omission]` |
| `filter.levels` | only these levels | `[P0, P2]` |
| `filter.evidence_conditions` | only these | `[none]` or `[k0, k3]` |
| `filter.rule_ids` | only these rules | `[S1, S3]` |
| `filter.where` | any record field, including rule tags | `{rule_tags.level: [3, 4]}` |
| `sample.per_cell` | counterfactual cases per (rule, type, evidence) cell; each is kept at all its levels, so levels stay comparable | `2` |
| `sample.seed` | which ones | `0` |
| `sample.limit` | at most this many records (quick tests) | `50` |
| `prompt.template` | `nsg_paper`, or a path to your own template YAML (copy `nsg_sim/prompts/templates/nsg_paper.yaml`) | |
| `prompt.instruction` | `follow` ("follow the nurse's reasoning") or `neutral` | `neutral` |
| `prompt.explanation_format` | `text`, or `formal` (the rule as a formula, P0 only) | `text` |
| `with_baseline` | also ask without the explanation; needed for NSG | `true` |
| `simulators` | one or more, see below | |
| `run.concurrency` | parallel requests | `4` |
| `run.max_tokens` | answer length limit | `2048` |
| `run.max_requests` | refuse to start if more requests would be needed | `1500` |
| `run.timeout` / `run.retries` | per request | `300` / `2` |

**Simulators**:

| Key | Meaning | Example |
| --- | --- | --- |
| `backend` | `ollama`, `openrouter`, `openai`, `vllm` (any OpenAI-compatible server), `baseline` | `ollama` |
| `model` | model name at that provider, or a baseline name | `gpt-oss:20b-cloud`, `openai/gpt-oss-20b`, `follow_g` |
| `reasoning_effort` | for reasoning models | `low`, `medium`, `high` |
| `temperature` | default 0 | `0.7` |
| `max_tokens` | overrides `run.max_tokens` | `4096` |
| `base_url` | overrides the address from `.env` | `http://MacBook-Pro-3.local:11434/v1` |
| `label` | name in the results (default `model@effort`) | `gptoss-low` |
| `params` | extra request fields for the provider | `{top_p: 0.9}` |

---

## Step 6 · Run

```bash
nsg run configs/experiments/triage_simple_openrouter.yaml --dry-run   # count first, nothing is sent
nsg run configs/experiments/triage_simple_openrouter.yaml
```

```
triage_simple_openrouter: 40 records x 2 prompt variants = 80 rows per simulator, 50 unique prompts
  openai/gpt-oss-20b@low [openrouter]: 80 rows to run (0 done), 50 API requests
```

- **Start small**: `sample: {per_cell: 1}` (one case per cell) or `limit`, look at the answers, then scale up.
- **Interrupting is safe**: run the same command again to finish the missing rows and retry
  failed ones.
- **Cost guard**: if a run would need more than `max_requests` requests, it does not start.
- **Not built yet?** `nsg run ... --build-missing` builds the dataset from `configs/datasets/` first.

Everything about a run lands in `runs/<name>/`:

| File | Content |
| --- | --- |
| `<simulator>.jsonl` | one row per answer: the parsed YES / NO, the raw response, the model's reasoning, time, tokens |
| `run.json` | the full experiment config, dataset hash, git commit, start time |
| `summary.csv` | accuracy, NSG and reliance with 95% intervals per simulator × explanation type × level × evidence |
| `nsg.png` | NSG by level, one line per explanation type, one panel per simulator (`acc.png` if `with_baseline: false`) |

`summary.csv` and the figure are written after every run (`--no-report` skips them);
`nsg report runs/<name>` rewrites them, e.g. for runs made before this existed.

---

## Step 7 · Results

```bash
nsg analyze runs/triage_simple_openrouter                                  # default grouping
nsg analyze runs/triage_simple_openrouter --by explanation_type,level
nsg analyze runs/triage_simple_openrouter --by explanation_type,rule_tags.n_conditions --where level=P0 --csv out.csv
nsg analyze runs/triage_simple_openrouter runs/triage_simple_counter_openrouter --by variant,explanation_type   # several runs, e.g. with a mirror twin
nsg plot runs/triage_simple_openrouter --metric nsg --x level --lines explanation_type --out figures/triage_nsg.png
```

| Column | Meaning |
| --- | --- |
| `acc` | accuracy with the explanation (95% interval in `acc_ci`) |
| `acc_wo` | accuracy without it |
| `nsg` | (acc − acc_wo) / (1 − acc_wo): how much the explanation helps (interval in `nsg_ci`) |
| `reliance` | on cases where the explanation is wrong: share answered the way it says |
| `invalid` | share of answers that could not be read as YES / NO (look at `raw_response`) |
| `n`, `clusters` | answers and reference patients behind the number |

Fields for `--by` and `--where`: `simulator`, `explanation` (type plus edit size, e.g.
`omission n=2`), `explanation_type`, `edit_size`, `level`,
`evidence_condition`, `rule_id`, `discriminating`, `variant`, `dataset`, `experiment`,
`label_cx`, and every rule tag as `rule_tags.<name>`. Intervals resample whole reference
patients, because their counterfactuals are not independent.

`nsg plot` options: `--metric nsg|acc|acc_wo|reliance`, `--x` (default `level`), `--lines`
(default `explanation`), `--panels` (default `simulator`), `--where`, `--title`, `--out`.

In Python or a notebook (`notebooks/playground.ipynb`):

```python
from nsg_sim.analysis.metrics import load_pairs, group, nsg, cluster_ci
pairs = load_pairs(["runs/triage_simple_openrouter"])
```

---

## Step 8 · Share

```bash
nsg push triage_simple_v1       # private repo spar-nsg-simulators/triage_simple_v1 on the HF hub
nsg pull triage_simple_v1       # teammates: into data/, checked against the hash
```

Commit the YAML files in `configs/`. `data/` and `runs/` are not in git: datasets are rebuilt
from their recipes or pulled from the hub.

---

## Going further

### Inputs that are not a table

For sentences, documents or anything where features are computed from the input, subclass
`nsg_sim.domain.base.Domain` and override four methods:

| Method | Returns |
| --- | --- |
| `sample_input(rng)` | a random input (any JSON-serialisable dict) |
| `perturb(x, rng, focus)` | a nearby input with 1–2 features changed, mostly those in `focus` |
| `feature_values(x)` | the feature values rules are evaluated on |
| `describe(x)` | the text shown to the simulator |

Put the class in a module, e.g. `nsg_sim/domain/sentences.py`, and point a domain YAML at it;
the rest of the YAML is passed to the class, so features and phrases are declared as in Step 1:

```yaml
class: nsg_sim.domain.sentences:SentenceDomain
name: sentences
features:
  n_adj: {type: numeric, range: [0, 3], label: number of adjectives}
words: {adj: [happy, small], noun: [cats, dogs], verb: [sleep, run], adv: [quietly, often]}
```

A complete working example (short sentences whose features are part-of-speech counts) is used
by the tests: `tests/sentence_domain.py` and `tests/fixtures/sentence_example.yaml`. The same pattern gives
**computed features** (a weighted score, a ratio of two features), since rules compare one feature
with a constant. The `mirror` and `abstract` transforms work on table domains only.

### Other additions (Python)

| To add | Where |
| --- | --- |
| a model provider | OpenAI-compatible: a preset in `nsg_sim/backends/openai_compat.py`; otherwise a class with `async complete()` and `@register("name")` in `nsg_sim/backends/` |
| an explanation type | `nsg_sim/explanations.py` and `EXPLANATION_TYPES` in `nsg_sim/config.py` |
| a metric | `METRICS` in `nsg_sim/analysis/metrics.py` |

---

## When something goes wrong

| Message | Cause | Fix |
| --- | --- | --- |
| `Extra inputs are not permitted` | a typo or an unknown key | check the key name against this guide |
| `rule S2: unknown feature 'temperatur'` | the feature is not in the domain | fix the name |
| `rule S4: arrival has no value 'car'` | category value not in `values` | use a listed value |
| `dataset 'x' already exists with different content` | the recipe or domain changed | new name: `x_v2` |
| `... was run on another version of x; use a new experiment name` | the dataset changed under a run | new experiment name |
| `N requests exceed run.max_requests` | the run is bigger than your cap | smaller `sample`, or raise the cap |
| `no rule with approval rate in [...]` | sampler settings too strict | widen `rule_yes_rate` / `condition_pass_rate`, allow more connectors |
| `domain 'x' not found` | no `configs/domains/x.yaml` | check the file name |
| many `invalid` answers | the model does not use the answer format | read `raw_response` in `runs/<name>/*.jsonl`, raise `max_tokens` |

---

## Checklist

1. Domain: labels, `>=` and `<` phrases with threshold lists, a readable profile.
2. Recipe: a name ending in `_v1`, rules, explanation types, counterfactuals.
3. `nsg validate` → `nsg preview` (read the texts) → `nsg build -v` (check the shares).
4. Baselines: `follow_g` NSG = 1 on faithful and below 1 on unfaithful types.
5. Experiment: `--dry-run`, small sample first, `max_requests` set.
6. `nsg analyze` / `nsg plot`.
7. `nsg push` if others need the dataset; commit the YAML files.
