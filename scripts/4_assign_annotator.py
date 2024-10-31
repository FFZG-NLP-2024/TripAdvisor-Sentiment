import random

# Load restaurants from the file
with open('data/selected_restaurants.txt', 'r') as file:
    restaurants = [line.strip() for line in file.readlines()]

# Shuffle the restaurants list to ensure randomness
random.shuffle(restaurants)

# Define the number of people and restaurants per person
num_people = 5
restaurants_per_person = 10

# Check if there are enough restaurants for each person
total_needed = num_people * restaurants_per_person
if total_needed > len(restaurants):
    raise ValueError("Not enough restaurants for each person to have 10.")

# Assign restaurants to each person
assignments = {}
for i in range(num_people):
    person_name = f"Person_{i + 1}"
    assignments[person_name] = restaurants[i * restaurants_per_person:(i + 1) * restaurants_per_person]

# Print assignments
for person, assigned_restaurants in assignments.items():
    print(f"{person}:")
    for restaurant in assigned_restaurants:
        print(f" - {restaurant}")
    print("\n")

# Write assignments to an output file
with open('data/assigned_restaurants.txt', 'w') as output_file:
    for person, assigned_restaurants in assignments.items():
        output_file.write(f"{person}:\n")
        for restaurant in assigned_restaurants:
            output_file.write(f" - {restaurant}\n")
        output_file.write("\n")

# Print confirmation
print("Assignments have been written to 'assigned_restaurants.txt'")
