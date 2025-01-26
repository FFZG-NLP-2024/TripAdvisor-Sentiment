# Training Dataset Analysis, Sampling and Splitting

The "TripAdvisor Review Rating" dataset (jniimi/tripadvisor-review-rating) consists of 201,295 rows with columns like 'hotel_id', 'user_id', 'title', 'text', 'overall', and others capturing various hotel review aspects. For sentiment analysis, we focused on the 'text' and 'overall' columns, with 'overall' representing the aggregated rating (the target variable) and 'text' containing the review content. Ratings for individual hotel categories, such as 'cleanliness' or 'rooms', were not included as they weren't relevant to the overall sentiment.

## Statistical Distribution

The statistical distribution of the 'overall' ratings showed an imbalance, with:
- 42.98% of reviews labeled with a rating of 5,
- 33.65% with a rating of 4,
- 14.02% with a rating of 3,
- 5.36% with a rating of 2, and
- 4.00% with a rating of 1.

## Balancing the Dataset

To address this imbalance, we randomly sampled 8,000 reviews from each rating category, resulting in a balanced dataset of 40,000 reviews, with each label represented equally.

## Data Split

After balancing, the dataset was split into:
- Training set: 30,400 reviews 
- Validation set: 16,000 reviews 
- Test set: 8,000 reviews

The final numbers were obtained by initially splitting the balanced dataset into 80% for training and validation, and 20% for testing. The training and validation sets were then split 75-25, maintaining consistent label distribution through stratified sampling at each stage.

The splits were saved as Parquet files (`train.parquet`, `validation.parquet`, `test.parquet`).


## Dataset Availability

Finally, the dataset was uploaded to the Hugging Face Hub under the repository name `nhull/tripadvisor-split-dataset-v2` for broader access.
