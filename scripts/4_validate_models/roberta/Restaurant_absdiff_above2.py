import pandas as pd

# File path for the restaurant mapped results CSV
input_csv_restaurant = "Restaurant_Test_Mapped_Results.csv"
output_csv_restaurant = "Restaurant_Test_absdiff_above2.csv"

# Load the CSV file
df_restaurant = pd.read_csv(input_csv_restaurant)

# Filter rows where the absolute difference is 3 or more
df_filtered_restaurant = df_restaurant[df_restaurant["Absolute Difference"] >= 2]

# Save the filtered data to a new CSV
df_filtered_restaurant.to_csv(output_csv_restaurant, index=False)

print(f"Filtered restaurant dataset saved as {output_csv_restaurant}")
