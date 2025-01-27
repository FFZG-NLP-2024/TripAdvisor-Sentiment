# import pandas as pd

# # Load the CSV file
# file_path = "/home/mkokalj/projects/test/Hotel_Test_instance_results_1.csv"
# data = pd.read_csv(file_path)

# # Filter rows where Correct == 0
# incorrect_predictions = data[data['Correct'] == 0].copy()

# # Calculate the absolute difference between Real Value and Predicted Value
# incorrect_predictions['Absolute Difference'] = (incorrect_predictions['Real Value'] - incorrect_predictions['Predicted Value']).abs()

# # Add a column for line numbers
# incorrect_predictions['Line Number'] = incorrect_predictions.index + 1  # Adjust for 1-based indexing

# # Print the result
# print("Incorrect Predictions with Absolute Differences and Line Numbers:")
# print(incorrect_predictions)

# # Save the results to a new CSV file
# incorrect_predictions.to_csv("Incorrect_Predictions_for hotels_Analysis_1.csv", index=False)





# import pandas as pd

# # Load the newly created file
# file_path = "Incorrect_Predictions_for hotels_Analysis_1.csv"
# data = pd.read_csv(file_path)

# # Filter rows where Absolute Difference >= 2
# filtered_data = data[data['Absolute Difference'] >= 2]

# # Save the filtered results to a new CSV file
# filtered_data.to_csv("Filtered_Predictions_for_hotels_Analysis_1.csv", index=False)

# # Print the filtered results
# print("Rows with Absolute Difference >= 2:")
# print(filtered_data)



import pandas as pd
from collections import Counter

# Load the filtered file
file_path = "Filtered_Predictions_for_hotels_Analysis_1.csv"
filtered_data = pd.read_csv(file_path)

# Count occurrences of each Absolute Difference directly from the filtered file
absolute_difference_counts = Counter(filtered_data['Absolute Difference'])

# Alternatively, use pandas' value_counts()
absolute_difference_counts_pd = filtered_data['Absolute Difference'].value_counts()

# Print the counts
print("\nCounts of Absolute Differences (using Counter):")
print(absolute_difference_counts)

print("\nCounts of Absolute Differences (using pandas):")
print(absolute_difference_counts_pd)



#RESULT
# Counts of Absolute Differences (using Counter):
# Counter({2: 125, 3: 10, 4: 6})

# Counts of Absolute Differences (using pandas):
# Absolute Difference
# 2    125
# 3     10
# 4      6
# Name: count, dtype: int64






