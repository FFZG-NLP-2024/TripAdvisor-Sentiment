import os
from bs4 import BeautifulSoup

# Path to the directory containing HTML files
directory_path = 'data/src'

# List to store all extracted restaurant data
all_restaurants = []

# Loop through all HTML files in the specified directory
for filename in os.listdir(directory_path):
    if filename.endswith('.htm') or filename.endswith('.html'):  # Check for HTML files
        file_path = os.path.join(directory_path, filename)
        
        # Load the content of the HTML file
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # Parse the file content with BeautifulSoup
        soup = BeautifulSoup(file_content, 'html.parser')

        # Extract restaurant names, locations, stars, and regions
        for restaurant in soup.find_all(attrs={"data-restaurant-name": True, "data-restaurant-country": True, "data-dtm-distinction": True, "data-dtm-region": True}):
            name = restaurant['data-restaurant-name']
            country = restaurant['data-restaurant-country']  # Renamed variable to country
            stars = restaurant['data-dtm-distinction']
            region = restaurant['data-dtm-region']  # Extracted region
            all_restaurants.append((name, country, region, stars))  # Updated order of tuple

# Remove duplicates for a unique list of restaurants and their details
unique_restaurants = list(set(all_restaurants))

# Save the unique list of restaurants and their details to a single text file
output_file_path = 'data/1_restaurants_list.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for name, country, region, stars in unique_restaurants:  # Adjusted unpacking to match new order
        file.write(f"{name} - {country} - {region} - {stars}\n")  # Updated output format to reflect the order

print(f"Extracted {len(unique_restaurants)} unique restaurants and saved to {output_file_path}.")
