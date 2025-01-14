import random
from collections import defaultdict

# Load data from the file
input_file_name = 'data/2_sample/1_restaurants_list.txt'
output_file_name = 'data/2_sample/2_selected_restaurants.txt'

# Step 1: Read the file and organize data by country code and star rating
data = []
with open(input_file_name, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Grouping the restaurants by star rating and country code
grouped_data = defaultdict(lambda: defaultdict(list))
for line in lines:
    parameters = [param.strip() for param in line.strip().split('-')]
    if len(parameters) > 1:
        country_code = parameters[1].strip()
        if len(parameters) > 3:
            # Re-join the restaurant name in case it contains a hyphen
            restaurant_name = ' - '.join(parameters[:-2]).strip()  # Join all parts except last two
            star_rating = parameters[-1].strip()  # Last part is the star rating
            data.append((restaurant_name, country_code, star_rating))  # Store (restaurant_name, country_code, star_rating)
            grouped_data[star_rating][country_code].append(restaurant_name)  # Group by star rating and country code

# Print grouped data
print("\nGrouped Data by Star Rating and Country Code:")
for star_rating, countries in grouped_data.items():
    print(f"{star_rating}:")
    for country_code, restaurants in countries.items():
        print(f"  {country_code}: {restaurants}")

# Step 2: Sample 20 restaurants for each star rating, ensuring country diversity and no duplicates
sampled_restaurants = {}
already_sampled = set()  # Set to keep track of already sampled restaurant names

for star_rating, countries in grouped_data.items():
    print(f"\nSampling for {star_rating} star:")
    sampled_restaurants[star_rating] = []
    
    # Track the countries we've already sampled from and their counts
    sampled_countries = defaultdict(int)  # To count restaurants per country
    
    while len(sampled_restaurants[star_rating]) < 20:
        # Randomly select a country and a restaurant from that country
        country_code = random.choice(list(countries.keys()))  # Randomly choose a country
        if countries[country_code]:  # Ensure there are restaurants available
            sampled_restaurant = random.choice(countries[country_code])  # Randomly choose one restaurant
            if sampled_restaurant not in already_sampled:  # Check for duplicates
                # Append restaurant name, country code, and star rating as a tuple
                sampled_restaurants[star_rating].append((sampled_restaurant, country_code, star_rating))
                already_sampled.add(sampled_restaurant)  # Mark this restaurant as sampled
                sampled_countries[country_code] += 1  # Increment count for this country
                print(f"Sampled from {country_code}: {sampled_restaurant}")

    # Print the number of unique countries sampled for this star rating
    print(f"Number of unique countries sampled for {star_rating} star: {len(sampled_countries)}")
    
    # Print how many were sampled from each country
    print(f"Counts from each country for {star_rating} star:")
    for country_code, count in sampled_countries.items():
        print(f"  {country_code}: {count}")

# Step 3: Save the sampled restaurants to a text file
with open(output_file_name, 'w', encoding='utf-8') as output_file:
    for star_rating, restaurants in sampled_restaurants.items():
        output_file.write(f"{star_rating} Star Restaurants:\n")
        for restaurant, country_code, stars in restaurants:
            output_file.write(f"{restaurant} - {country_code} - {stars}\n")  # Write restaurant, country code, and stars
        output_file.write("\n")  # Add a blank line for separation

print(f"Sampled restaurants saved to {output_file_name}.")

# Print the final sampled restaurants
print("\nFinal Sampled Restaurants:")
for star_rating, restaurants in sampled_restaurants.items():
    print(f"\n{star_rating} Star Restaurants:")
    for restaurant, country_code, stars in restaurants:
        print(f"{restaurant} - {country_code} - {stars}")