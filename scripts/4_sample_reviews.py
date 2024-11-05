import pandas as pd
import nltk
import re
import numpy as np

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Load your reviews data
df = pd.read_csv('data/3_reviews_no_line_breaks.csv')

# Function to count characters in reviews
def count_characters(df):
    original_reviews = []
    char_counts = []  # List to store character counts
    other_columns = []

    # Get all original columns except 'Content'
    original_columns = df.columns.tolist()
    original_columns.remove('Content')

    # Use 'Content' as the column name for reviews
    for _, row in df.iterrows():  # Iterate over DataFrame rows
        review = row['Content']
        
        # Count characters (Latin alphanumeric signs without line breaks)
        char_count = len(re.findall(r'[A-Za-z0-9]', review.replace('\n', '')))  # Remove line breaks and count

        # Append the original review and its character count
        original_reviews.append(review)
        char_counts.append(char_count)
        
        # Add the other column values for the review
        other_columns.append([row[col] for col in original_columns])
    
    # Create a new DataFrame with original reviews and character counts
    result_df = pd.DataFrame({
        'original_review': original_reviews,
        'char_count': char_counts  # Add character counts to DataFrame
    })

    # Add the other columns to the result DataFrame
    for i, col in enumerate(original_columns):
        result_df[col] = [item[i] for item in other_columns]

    # Add an index column for each review
    result_df.insert(0, 'index', range(1, len(result_df) + 1))  # Start indexing from 1

    return result_df

# Apply the function
char_count_df = count_characters(df)

# Calculate the average character count
average_length = char_count_df['char_count'].mean()
print(f"Average Character Count: {average_length:.2f}")

# Define a range around the average length (e.g., +/- 50 characters)
lower_bound = max(0, average_length - 50)  # Ensure lower bound is not negative
upper_bound = average_length + 50

# Filter the DataFrame for reviews within this range
filtered_reviews = char_count_df[(char_count_df['char_count'] >= lower_bound) & (char_count_df['char_count'] <= upper_bound)]

# Randomly sample 25 reviews from the filtered DataFrame
sampled_reviews = filtered_reviews.sample(n=min(25, len(filtered_reviews)), random_state=1)

# Add a new index for the sampled reviews
sampled_reviews.insert(0, 'sample_index', range(1, len(sampled_reviews) + 1))  # Start indexing from 1

# Save the sampled reviews to a new CSV file
sampled_reviews.to_csv('data/4_sampled_reviews_around_average.csv', index=False)

# Print the sampled reviews
print("Sampled Reviews Around Average Character Count:")
print(sampled_reviews[['sample_index', 'index', 'original_review', 'char_count']])
