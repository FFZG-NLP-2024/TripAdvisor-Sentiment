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

        # Extract restaurant names and locations
        for restaurant in soup.find_all(attrs={"data-restaurant-name": True, "data-restaurant-country": True}):
            name = restaurant['data-restaurant-name']
            location = restaurant['data-restaurant-country']
            all_restaurants.append((name, location))

# Remove duplicates for a unique list of restaurants and their locations
unique_restaurants = list(set(all_restaurants))

# Save the unique list of restaurants and their locations to a single text file
output_file_path = 'data/combined_restaurants1.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for name, location in unique_restaurants:
        file.write(f"{name} - {location}\n")

print(f"Extracted {len(unique_restaurants)} unique restaurants and saved to {output_file_path}.")
