# SPAR NSG

## Datasets:

Datasets are stored at our hf organization: [SPAR NSG Simulators](https://huggingface.co/spar-nsg-simulators)

## Quick Start:

### Clone and access repo:

```bash
git clone https://github.com/spar-nsg-team/nsg-evaluation.git
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
