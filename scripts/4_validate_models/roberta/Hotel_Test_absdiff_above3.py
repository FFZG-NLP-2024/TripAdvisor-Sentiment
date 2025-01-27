import pandas as pd

# File path for the hotel mapped results CSV
input_csv_hotel = "Hotel_Test_Mapped_Results.csv"
output_csv_hotel = "Hotel_Test_absdiff_above3.csv"

# Load the CSV file
df_hotel = pd.read_csv(input_csv_hotel)

# Filter rows where the absolute difference is 3 or more
df_filtered_hotel = df_hotel[df_hotel["Absolute Difference"] >= 3]

# Save the filtered data to a new CSV
df_filtered_hotel.to_csv(output_csv_hotel, index=False)

print(f"Filtered hotel dataset saved as {output_csv_hotel}")
