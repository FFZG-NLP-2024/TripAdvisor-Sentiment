from datasets import load_dataset

# Load the dataset
dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")

# Save each split (train, validation, test) to CSV files
for split in dataset:
    dataset[split].to_csv(f"tripadvisor_{split}.csv", index=False)

print("Dataset saved as CSV files.")
