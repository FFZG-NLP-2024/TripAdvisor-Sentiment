# Report 1: Data Acquisition and Review Processing

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

4. **Assigning Restaurants to Group Members**: To ensure an organized approach to data verification, each of the five group members is assigned a unique set of 10 restaurants from the `selected_restaurants.txt` file. A Python script (`4_assign_annotator.py`) performs this assignment as follows:
   - **Data Loading**: It reads the list of selected restaurants.
   - **Random Assignment**: The script shuffles the list and assigns 10 unique restaurants to each placeholder (`Person_1` through `Person_5`).
   - **Alphabetical Assignment of Members**: Finally, group members were assigned to each placeholder (`Person_1`, `Person_2`, etc.) according to the alphabetical order of their last names—the first surname alphabetically was assigned to `Person_1`, the second to `Person_2`, and so on.
   - **Output**: The assignments for each group member are saved to a file (`assigned_restaurants.txt`), providing each person with their list of restaurants to verify.

5. **Getting Reviews**: After assignment, each group member manually verifies the presence of their assigned restaurants on TripAdvisor to ensure our dataset's reliability. We anticipated most restaurants would be listed on TripAdvisor and have a sufficient number of English-language reviews, which was one of our criteria for review selection. However, during verification, we encountered some issues. Although most restaurants were present on TripAdvisor, some either had no reviews or fewer than 25 English-language reviews, which is the maximum allowed by the free version of our Chromium extension. Despite these limitations, we decided to keep these restaurants in our list and proceed with the available reviews. If the number of reviews proves insufficient in the next phase, we will either add reviews manually or consider alternative solutions. A new script, (`3_clear_reviews.py`), has been implemented to process reviews stored in the (`data/src/reviews`) folder, which contains initially scraped reviews using a Chromium extension.

Since our group consists of five members, it was essential to distribute the review data among us. The folders are organized so that each member gets their own subset of reviews in their own folder (`data/ cleared_reviews`).

Once this phase is complete, these individual review files will be merged into a single, comprehensive file in the main data folder, making it easier to proceed with annotation.

## Review Processing

Following the data acquisition phase, the next significant part of our project involves working with the reviews gathered from TripAdvisor. This includes assigning sentiment to the reviews and developing annotation guidelines to standardize our analysis process:
   - **Assigning Sentiment**: The first step involves assigning a sentiment to a smaller subset of reviews. This is done manually by each group member. The results are stored into the `data/sentiment/1_sample_assignment` folder (different Excel files for each person (`person_1_removed_line_breaks.xslx` etc.)). These Excel files were created manually in Excel by going to *Data* > *Get Data* > *From File* > *From Text/CSV*, then selecting files without line breaks from the `data/cleared_reviews` folder for each person.
   - **Annotation Guidelines**: The second step involves creating an annotation guideline for each review.

## Conclusion

The scripts and organization of this project facilitate efficient extraction, selection, and sentiment annotation of Michelin-starred restaurant reviews. Through a combination of automated data extraction and structured manual verification, we aim to build a robust and reliable dataset for sentiment analysis.

---

**Note**: This report is a living document and will be updated continuously to reflect the project’s current phase and progress. Future additions will detail ongoing modifications, insights from the sentiment analysis, and further developments in review processing and data interpretation.