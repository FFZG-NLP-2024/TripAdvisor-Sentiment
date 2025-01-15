from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd

# Check if the dataset is already loaded in the environment (adjust as necessary)
dataset_name = "jniimi/tripadvisor-review-rating"

# Load the dataset only if it's not already loaded in memory
try:
    dataset = load_dataset(dataset_name)
except:
    print(f"Could not load dataset {dataset_name}. Please check the source.")
    raise

# Extract the columns of interest: 'review' and 'overall'
df = pd.DataFrame(dataset['train'])

# Rename 'overall' column to 'label'
df = df[['review', 'overall']].rename(columns={'overall': 'label'})

# Balance the dataset by limiting to 8,000 reviews per label
balanced_df = df.groupby('label').apply(lambda x: x.sample(n=8000, random_state=42)).reset_index(drop=True)

# Split into train+validation (80%) and test (20%)
train_val, test = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['label'])

# Split train+validation into train (75% of train+validation) and validation (25% of train+validation)
train, validation = train_test_split(train_val, test_size=0.05, random_state=42, stratify=train_val['label'])

# Save as Parquet files
train.to_parquet('dataset/train.parquet', index=False)
validation.to_parquet('dataset/validation.parquet', index=False)
test.to_parquet('dataset/test.parquet', index=False)

# Print out the number of examples in each set
print(f"Number of examples in train set: {len(train)}")
print(f"Number of examples in validation set: {len(validation)}")
print(f"Number of examples in test set: {len(test)}")

# Print out the total number of examples
total_examples = len(train) + len(validation) + len(test)
print(f"Total number of examples across all sets: {total_examples}")

print("Dataset split and saved as Parquet files successfully.")
