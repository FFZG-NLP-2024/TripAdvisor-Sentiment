import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Load the CSV file
file_path = 'data/4_sampled_reviews_around_average.csv'
data = pd.read_csv(file_path)

# Download the punkt tokenizer if needed
nltk.download('punkt')

# Create a list to store results
sentence_data = []

# Process each review
for _, row in data.iterrows():
    review_index = row['sample_index']
    review_text = row['original_review']
    
    # Split review into sentences
    sentences = sent_tokenize(review_text)
    
    # Enumerate sentences with a sub-index format: review_index.sentence_number
    for i, sentence in enumerate(sentences, 1):
        # Create a new entry for each sentence
        sentence_entry = {
            'review_index': review_index,
            'sentence_index': f"{review_index}.{i}",
            'sentence': sentence.strip(),
            'original_review': review_text  # Include the full review content
        }
        
        # Add all original columns (excluding 'original_review' to avoid duplication)
        for col in row.index:
            if col != 'original_review':
                sentence_entry[col] = row[col]
        
        sentence_data.append(sentence_entry)

# Convert the list to a DataFrame
sentence_df = pd.DataFrame(sentence_data)

# Save the new DataFrame to a CSV file
output_path = 'data/5_reviews_sentences.csv'
sentence_df.to_csv(output_path, index=False)

print("Sentences with full content have been saved to:", output_path)