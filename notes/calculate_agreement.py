import pandas as pd
from scipy.stats import mode
import numpy as np  # Added for handling array-like structures

# Load the data from the Excel file
# file_path = "/Users/meta/Desktop/PITCH/25_sample_reviews_full.xlsx"  # Replace with the correct file path
df = pd.read_excel(file_path)

# Define the columns that contain annotator ratings for each sentence
annotator_columns = [
    "Sentiment (sentence) 1",
    "Sentiment (sentence) 2",
    "Sentiment (sentence) 3",
    "Sentiment (sentence) 4",
    "Sentiment (sentence) 5"
]


# Calculate sentence-level agreement
def calculate_agreement(row):
    labels = row[annotator_columns]
    # Extract the rating number from strings like "5 (positive)"
    ratings = [int(label.split()[0]) for label in labels if isinstance(label, str)]
    if len(ratings) == 0:
        return None
    most_common, count = mode(ratings)

    # Handle cases where mode() might return a scalar instead of an array
    if isinstance(count, (list, tuple, pd.Series, np.ndarray)):
        agreement_level = count[0] / len(ratings)
    else:
        agreement_level = count / len(ratings)  # Handle scalar count

    return agreement_level


# Apply agreement calculation to each sentence
df["Sentence_Agreement"] = df.apply(calculate_agreement, axis=1)

# Calculate paragraph-level agreement by grouping sentences by `review_index`
# and averaging Sentence_Agreement within each paragraph
df["Paragraph_ID"] = df["review_index"]  # Adjust if necessary to use the correct paragraph ID column
paragraph_agreement = df.groupby("Paragraph_ID")["Sentence_Agreement"].mean()
df["Paragraph_Agreement"] = df["Paragraph_ID"].map(paragraph_agreement)

# Calculate the difference in agreement between each sentence and its paragraph
df["Agreement_Difference"] = df["Sentence_Agreement"] - df["Paragraph_Agreement"]

# Save the results to a new Excel file for analysis
output_file_path = "agreement_analysis_with_paragraphs.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Agreement analysis saved to '{output_file_path}'")

