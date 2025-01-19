import os
import pandas as pd

# Replace with the actual paths to your Parquet files
train_file = "scripts/2_split_dataset/train.parquet"
validation_file = "scripts/2_split_dataset/validation.parquet"
test_file = "scripts/2_split_dataset/test.parquet"

# Get file sizes in bytes
train_size = os.path.getsize(train_file)
validation_size = os.path.getsize(validation_file)
test_size = os.path.getsize(test_file)

# Load Parquet files to count the number of examples
train_df = pd.read_parquet(train_file)
validation_df = pd.read_parquet(validation_file)
test_df = pd.read_parquet(test_file)

# Get number of examples
train_examples = len(train_df)
validation_examples = len(validation_df)
test_examples = len(test_df)

# Print the results
print(f"Train file: {train_size} bytes, {train_examples} examples")
print(f"Validation file: {validation_size} bytes, {validation_examples} examples")
print(f"Test file: {test_size} bytes, {test_examples} examples")