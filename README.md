# SPAR NSG

## Datasets:

Datasets are stored at our hf organization: [SPAR NSG Simulators](https://huggingface.co/spar-nsg-simulators)

## Quick Start:

### Clone and access repo:

```bash
git clone https://github.com/karolrogozinski/spar-nsg-simulators.git
cd spar-nsg-simulators

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Login to access private datasets:
```bash
python3 -c "from huggingface_hub import login; login()"
```
_(At least the above is the one that was working for me)_

## Pipeline (`nsg_sim`)

Datasets and experiments are YAML files in `configs/`, built and run with the `nsg` command:

```bash
pip install -e .
nsg preview configs/datasets/triage_simple_v1.yaml     # read the rules and explanation texts
nsg build configs/datasets/triage_simple_v1.yaml       # -> data/triage_simple_v1/
nsg run configs/experiments/triage_simple_smoke.yaml   # -> runs/triage_simple_smoke/ (free baselines)
nsg analyze runs/triage_simple_smoke --by simulator,explanation_type
pytest -q
```

Step-by-step guide with every option: [docs/ADDING_A_DATASET.md](docs/ADDING_A_DATASET.md).

API keys and endpoints go in a git-ignored `.env` file in the repo root (see Step 0 of the guide).

### What is in `configs/`

| Files | What they are |
| --- | --- |
| `domains/<name>.yaml` | 7 tabular domains with a known decision rule, from strong simulator priors to none: `triage_v3`, `loan`, `irrigation`, `beekeeping`, `stars`, `library`, `shells` (`triage.yaml` is the older triage version) |
| `datasets/<domain>_sim_v1.yaml` | intuitive rules, faithful and omission explanations at four precision levels P0–P3 |
| `datasets/<domain>_sim_counter_v1.yaml`, `..._abstract_v1.yaml` | the same records with a counter-intuitive rule (`mirror`) or with abstract feature names (`abstract`); labels are identical across the three twins |
| `datasets/triage_evidence_v1.yaml` | labelled examples in the prompt |
| `experiments/sim_e8_*`, `sim_e10_*`, `sim_e11_*` | simulator runs: stronger simulators and reasoning, examples, the 7-domain comparison |
| `experiments/priors_*.yaml` | simulator priors: P(YES) on cases with no rule and no explanation |

Datasets are rebuilt identically from their recipes (`nsg build`, checked against the hash in
`manifest.json`), so `data/` and `runs/` are not in git.

### Acknowledgements
This project builds on the code of [Mayne et al., 2026](https://arxiv.org/abs/2602.02639),
available at https://github.com/HarryMayne/faithfulness under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
