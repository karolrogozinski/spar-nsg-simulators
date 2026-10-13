# SPAR NSG

## Datasets:

Datasets are stored at our hf organization: [SPAR NSG Simulators](https://huggingface.co/spar-nsg-simulators)

## Quick Start:

### Clone and access repo:

```bash
git clone https://github.com/spar-nsg-team/spar-nsg-evaluation.git
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

### Acknowledgements
This project builds on the code of [Mayne et al., 2026](https://arxiv.org/abs/2602.02639),
available at https://github.com/HarryMayne/faithfulness under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
