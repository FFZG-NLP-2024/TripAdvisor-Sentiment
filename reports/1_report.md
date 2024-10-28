# Report 1: Sentiment Analysis of Michelin Restaurants - Data Acquisition and Review Processing

## Background

In our project, we aim to analyze the sentiment of Michelin-starred restaurants that are marked as sustainable. This selection criterion was chosen because it provides a reliable dataset for our analysis. Initially, we considered focusing on restaurants across Europe and specific regions; however, the official Michelin website does not allow filtering by location—only by cuisine type. This limitation prompted us to adjust our approach and focus solely on extracting all relevant restaurants from the Michelin site.

## Data Acquisition

The first part of our project involves a structured approach to data acquisition, divided into four main steps:

1. **Setting Criteria**: We defined our criteria to include only Michelin-starred restaurants that are recognized for their sustainable practices.

2. **Getting All Restaurants**: The second step involved gathering all restaurants that match our criteria. We utilized a Python script named `1_get_restaurants.py` that performs the following tasks:
   - **Directory Traversal**: The script searches a specified directory (`src/data`) for HTML files containing restaurant data.
   - **HTML Parsing**: It uses the BeautifulSoup library to parse each HTML file and extract restaurant names and their corresponding locations, which are stored as tuples in a list.
   - **Duplicate Removal**: After extracting data from all files, the script eliminates duplicates to ensure a unique list of restaurants.
   - **Output**: The final unique list of restaurants and their locations is saved to a text file (`combined_restaurants.txt`) for further processing. The script also provides a summary of the number of unique restaurants extracted.

3. **Random Selection**: The third step involves selecting a balanced random sample of restaurants from the combined list. The second script, `2_restaurants_selection.py`, performs the following tasks:
   - **Data Loading**: It reads the previously generated `combined_restaurants.txt` file, grouping the restaurants by their respective countries.
   - **Proportional Selection**: The script calculates the proportion of restaurants from each country and uses this information to determine how many restaurants to select from each country, ensuring diversity in the sample.
   - **Random Sampling**: A total of 50 restaurants are randomly selected, with the selection process accounting for the distribution of restaurants by country.
   - **Output**: The selected restaurants, along with their countries, are saved to a new text file (`selected_restaurants.txt`).

4. **Getting Reviews**: After obtaining the selected restaurants, each of the five group members will manually verify the presence of these restaurants on TripAdvisor, with each member checking 10 restaurants. This verification is crucial for ensuring the reliability of our dataset before proceeding to the next phase.

## Review Processing

Following the data acquisition phase, the next significant part of our project involves working with the reviews gathered from TripAdvisor. This includes assigning sentiment to the reviews and developing annotation guidelines to standardize our analysis process. A new script, (`3_clear_reviews.py`), has been implemented to process reviews stored in the (`data/src/reviews`) folder, which contains initially scraped reviews using a Chromium extension. The filtered reviews are then combined into a single output file in the (`data`) folder.

## Conclusion

The scripts developed for this project facilitate the efficient extraction and selection of relevant restaurant data. By combining automated data extraction with manual verification, we aim to build a robust dataset for our sentiment analysis, focusing on reliable and pertinent information.
