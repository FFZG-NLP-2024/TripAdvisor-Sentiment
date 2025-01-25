import csv

# Function to read multiline CSV
def read_multiline_csv(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append(row)
    return rows

# Load the CSV file
rows = read_multiline_csv('tripadvisor_train.csv')

# Calculate the average length of the reviews
total_length = 0
for row in rows:
    review = row['review'].strip()  # Trim leading and trailing whitespace
    review = review.replace('\n', '')  # Remove newline characters
    total_length += len(review)

# Average length calculation
average_length = total_length / len(rows)

# Print the first 5 reviews and their character counts
for i, row in enumerate(rows[:5]):
    review = row['review'].strip().replace('\n', '')  # Consistent trimming and newline removal
    print(f"Review {i + 1}: {review}")
    print(f"Length: {len(review)} characters\n")

print(f'The average length of the reviews is: {average_length:.2f} characters')