import random
from collections import defaultdict

# Load the combined restaurants data
input_file_path = 'src/data/combined_restaurants.txt'

# Dictionary to hold restaurants grouped by country
restaurants_by_country = defaultdict(list)

# Read the combined restaurants file and populate the dictionary
with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if ' - ' in line:
            try:
                name, location = line.rsplit(' - ', 1)
                restaurants_by_country[location].append(name)
            except ValueError:
                print(f"Skipping line due to formatting issue: {line}")
        else:
            print(f"Skipping line due to missing separator: {line}")

# Calculate the total number of restaurants
total_restaurants = sum(len(restaurants) for restaurants in restaurants_by_country.values())

# Calculate the proportion of restaurants from each country
country_proportions = {country: len(restaurants) / total_restaurants for country, restaurants in restaurants_by_country.items()}

# Determine the number of restaurants to select from each country
target_count = 50
selected_restaurants = []

# First pass: select based on proportion but keep track of total selections
for country, proportion in country_proportions.items():
    # Calculate how many to select based on proportion
    count_to_select = max(1, int(proportion * target_count))
    
    available_restaurants = restaurants_by_country[country]
    if available_restaurants:
        selected_from_country = random.sample(available_restaurants, min(count_to_select, len(available_restaurants)))
        # Store the restaurant with its country
        selected_restaurants.extend((restaurant, country) for restaurant in selected_from_country)

# If the total is still less than the target, fill the gap
if len(selected_restaurants) < target_count:
    all_available_restaurants = [(name, loc) for loc, names in restaurants_by_country.items() for name in names]
    additional_count = target_count - len(selected_restaurants)
    additional_selection = random.sample(all_available_restaurants, additional_count)
    selected_restaurants.extend(additional_selection)

# Shuffle the final selection for randomness
random.shuffle(selected_restaurants)

# Ensure the final selection is limited to 50
final_selection = selected_restaurants[:target_count]

# Save the selected restaurants to a new text file, including the country
output_file_path = 'src/data/selected_restaurants.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for restaurant, country in final_selection:
        file.write(f"{restaurant} - {country}\n")

print(f"Selected {len(final_selection)} restaurants and saved to {output_file_path}.")
