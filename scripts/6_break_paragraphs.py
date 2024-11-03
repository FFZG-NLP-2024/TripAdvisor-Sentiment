import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Load your reviews data
df = pd.read_csv('data/cleared_reviews/all_reviews/merged_reviews.csv')

# Print the column names to verify
print("Columns in DataFrame:", df.columns)

# Function to split reviews into sentences
def split_reviews(df):
    sentence_list = []
    original_reviews = []
    other_columns = []

    # Get all original columns except 'Content'
    original_columns = df.columns.tolist()
    original_columns.remove('Content')

    # Use 'Content' as the column name for reviews
    for _, row in df.iterrows():  # Iterate over DataFrame rows
        review = row['Content']
        # Split the review into sentences
        sentences = sent_tokenize(review)
        
        # Add each sentence to the sentence list
        sentence_list.extend(sentences)
        
        # Append the original review for later sentiment analysis
        original_reviews.extend([review] * len(sentences))  # Repeat the original review for each sentence
        
        # Add the other column values for each sentence
        for sentence in sentences:
            other_columns.append([row[col] for col in original_columns])
    
    # Create a new DataFrame with sentences and original reviews, including other columns
    result_df = pd.DataFrame({
        'sentence': sentence_list,
        'original_review': original_reviews
    })

    # Add the other columns to the result DataFrame
    for i, col in enumerate(original_columns):
        result_df[col] = [item[i] for item in other_columns]

    # Add an index column for each sentence
    result_df.insert(0, 'index', range(1, len(result_df) + 1))  # Start indexing from 1

    return result_df

# Apply the function
sentences_df = split_reviews(df)

# Save the sentences DataFrame to a new CSV file if needed
sentences_df.to_csv('data/cleared_reviews/all_reviews/split_sentences.csv', index=False)
