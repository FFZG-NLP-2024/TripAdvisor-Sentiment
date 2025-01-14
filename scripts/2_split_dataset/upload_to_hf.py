from datasets import DatasetDict

# Load your dataset from disk
dataset = DatasetDict.load_from_disk("dataset/split_tripadvisor_dataset")

# Push the dataset to the Hugging Face Hub
dataset.push_to_hub("nhull/tripadvisor-split-dataset", private=False)
