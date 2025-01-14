import pandas as pd
import os
import glob

# Define the input folder and output paths
input_folder = 'data/src/reviews/2_sample_150_reviews/*.csv'
output_path = "data/2_sample/3_reviews_cleared.csv"
output_path_cleaned = "data/2_sample/3_reviews_no_line_breaks.csv"

# Create an empty list to store DataFrames
all_reviews = []

# Iterate over all CSV files in the input folder
for file_path in glob.glob(input_folder):
    if os.path.exists(file_path):
        print(f"Processing file: {file_path}")
        
        # Load the data from the current CSV file
        df = pd.read_csv(file_path)
        
        # Extract the columns of interest: 'Published Date', 'Display Name', 'Review Text', 'URL', 'Rating', and 'Language'
        extracted_df = df[['Published Date', 'Display Name', 'Review Text', 'URL', 'Rating', 'Language']].copy()
        
        # Rename columns for clarity
        extracted_df.columns = ['Date', 'Author', 'Content', 'URL', 'Rating', 'Language']
        
        # Remove line breaks from the 'Content' column to ensure each review stays in a single row
        extracted_df.loc[:, 'Content'] = extracted_df['Content'].replace({'\n': ' ', '\r': ' '}, regex=True)
        
        # Append the DataFrame to the list
        all_reviews.append(extracted_df)
    else:
        print(f"File not found: {file_path}")

# Combine all DataFrames into a single DataFrame
combined_reviews = pd.concat(all_reviews, ignore_index=True)

# Save the combined data to a new CSV file
combined_reviews.to_csv(output_path, index=False)

# Save the cleaned data to a new CSV file
combined_reviews.to_csv(output_path_cleaned, index=False)

print(f"Combined reviews saved to: {output_path}")
print(f"Cleaned reviews saved to: {output_path_cleaned}")
