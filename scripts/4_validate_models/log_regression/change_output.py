import pandas as pd

# Path to the input file
input_file = "scripts/validate_models/log_regression/validation_results_log_regression.csv"
output_file = "scripts/validate_models/log_regression/updated_validation_results_log_regression.csv"

# Load the CSV file
data = pd.read_csv(input_file)

# Ensure the required columns exist
required_columns = ['Review', 'True Label', 'Predicted Label', 'Difference', 'Absolute Difference']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in the input file: {', '.join(missing_columns)}")

# Convert 'True Label' and 'Predicted Label' to integers and apply the renumbering
label_mapping = {6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 1: 0}

data['True Label'] = data['True Label'].astype(int).map(label_mapping)
data['Predicted Label'] = data['Predicted Label'].astype(int).map(label_mapping)

# Ensure 'Difference' and 'Absolute Difference' columns remain integers
data['Difference'] = data['Difference'].astype(int)
data['Absolute Difference'] = data['Absolute Difference'].astype(int)

# Save the updated DataFrame to a new file
data.to_csv(output_file, index=False)

print(f"Updated file saved to: {output_file}")
