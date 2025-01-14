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

# Split into train (80%), validation (15%), test (5%)
train, temp = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['label'])
validation, test = train_test_split(temp, test_size=0.25, random_state=42, stratify=temp['label'])

# Convert pandas DataFrames to HuggingFace Dataset objects
train_dataset = Dataset.from_pandas(train)
validation_dataset = Dataset.from_pandas(validation)
test_dataset = Dataset.from_pandas(test)

# Combine the datasets into one dataset dict
final_dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

# Save the resulting dataset to disk (you can later upload it to HuggingFace)
final_dataset.save_to_disk("scripts/dataset/split_tripadvisor_dataset")

print("Dataset split and saved successfully.")
