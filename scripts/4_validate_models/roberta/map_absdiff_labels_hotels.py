import pandas as pd
from datasets import load_dataset

# File paths
input_csv = "Hotel_Test_instance_results_1.csv"  # Input file with header in row 1
output_csv = "Hotel_Test_Mapped_Results.csv"  # Output file

# Load the hotel dataset (test part)
hotel_dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")
test_dataset = hotel_dataset["test"]

# Read the CSV and skip the first row (the header is in row 1)
df = pd.read_csv(input_csv, skiprows=1, names=["Real Value", "Predicted Value", "Correct"])

# Drop any empty rows
df = df.dropna()

# Prepare the output list
output_data = []

# Iterate through the rows in the dataframe
for idx, row in df.iterrows():
    # Get the corresponding review text from the test dataset
    review_text = test_dataset[idx]["review"]
    
    # Get the true label from the dataset
    true_label = row["Real Value"]
    
    # Get the predicted label from the CSV
    predicted_label = row["Predicted Value"]
    
    # Calculate the absolute difference between the true and predicted labels
    absolute_difference = abs(true_label - predicted_label)
    
    # Append the data for the new CSV
    output_data.append({
        "Review": review_text,
        "True Label": true_label,
        "Predicted Label": predicted_label,
        "Absolute Difference": absolute_difference
    })

# Create a DataFrame for the output data
output_df = pd.DataFrame(output_data)

# Save the output to a new CSV
output_df.to_csv(output_csv, index=False)

print(f"Mapped CSV saved as {output_csv}")

