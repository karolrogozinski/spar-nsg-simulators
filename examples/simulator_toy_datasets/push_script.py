from datasets import load_dataset

dataset = load_dataset("json", data_files="test_toy_dataset.jsonl")
dataset.push_to_hub("spar-nsg-simulators/test-to-dataset", private=True)
