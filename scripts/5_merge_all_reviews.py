import pandas as pd
import glob
import os

# Path where your CSV files are located
path = 'data/cleared_reviews/all_reviews'  # Change this to your path
all_files = glob.glob(os.path.join(path, "*.csv"))  # Get all CSV files

# List to store dataframes
dataframes = []

for file in all_files:
    # Read each CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Add a new column with the filename (without the path and extension)
    df['source_file'] = os.path.splitext(os.path.basename(file))[0]
    
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all dataframes into one big dataframe
merged_df = pd.concat(dataframes, ignore_index=True)

# Specify the full path for the new CSV file in the same folder
merged_file_path = os.path.join(path, 'merged_reviews.csv')

# Save the merged DataFrame to the specified path
merged_df.to_csv(merged_file_path, index=False)
