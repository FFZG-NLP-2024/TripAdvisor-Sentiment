## Overview

This is a group project conducted by Digital Linguistics students from the Faculty of Humanities and Social Sciences, University of Zagreb, as part of the NLP course. The project focuses on performing sentiment analysis of TripAdvisor reviews for Michelin-starred restaurants that are marked as sustainable. By concentrating on these criteria, we aim to obtain a reliable and unbiased dataset for our analysis.

The project is divided into two main parts: data acquisition and review processing.

The first part includes the following steps:
1. **Setting Criteria**: Defining the filters for selecting restaurants, focusing on Michelin stars and sustainability.
2. **Getting All Restaurants**: Extracting restaurant data from HTML files sourced from the official Michelin website.
3. **Random Selection**: Randomly selecting a balanced sample of 50 restaurants from the compiled list.
4. **Getting Reviews**: Scraping reviews for each randomly selected restaurant from TripAdvisor.

Once the data is gathered, the second part involves working with the reviews, assigning sentiment, and creating annotation guidelines for analysis.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [System Information](#system-information)
  - [Cloning the Repository](#cloning-the-repository)
  - [Installation](#installation)
  - [Running the Scripts](#running-the-scripts)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Credits](#credits)

## Getting Started

### Prerequisites
- **Python**: This project requires Python 3.7 or later.
- **Libraries**: The following Python libraries are needed:
  - `beautifulsoup4`
  - `pandas`

This section will be continuously updated.

You can install the required libraries using the `requirements.txt` file provided in the repository.

### System Information
- **Operating System:** Windows 11
- **Version:** 10.0.22631.4317 (Windows 11)
- **Windows Subsystem for Linux (WSL):** Enabled
- **WSL Distribution:** Ubuntu
- **WSL Version:** 2

### Cloning the Repository
To clone this repository, run the following command:

```bash
git clone https://github.com/FFZG-NLP-2024/TripAdvisor-Sentiment.git
```

### Installation
To install the required dependencies, navigate to the project directory and run:

```bash
pip install -r requirements.txt
```

### Running the Scripts

The project includes two main scripts for data acquisition:

    1_get_restaurants.py: This script extracts restaurant data from HTML files stored in the data/src/ directory and saves it to combined_restaurants.txt.
    2_restaurants_selection.py: This script selects a balanced random sample of 50 restaurants from the combined list and saves them to selected_restaurants.txt.
    3_clear_reviews.py: This script cleans the review data by removing line breaks and consolidating review information from multiple CSV files into a single CSV file.
    4_assign_annotator.py: This script assigns an annotator person to each restaurant in the selected_restaurants.txt file.

To run the scripts, navigate to the project directory in your terminal and execute the following commands:

```bash
scripts/1_get_restaurants.py
scripts/2_restaurants_selection.py
scripts/3_clear_reviews.py
scripts/4_assign_annotator.py
```

## Repository Structure

```bash
TripAdvisor-Sentiment
│
├── data/                                 # Data files
│   ├── src/                              # Source folder for HTML files
│   │   └── reviews/                      # Source folder for CSV files (per person)
│   ├── sentiment/                        # Manually annotated data
│   │   └── 1_sample_assignment/          # Manually annotated data (first phase)
│   ├── cleared_reviews_per_person/       # Output from 3_clear_reviews.py (per person)
│   ├── combined_restaurants.txt          # Output from 1_get_restaurants.py
│   ├── selected_restaurants.txt          # Output from 2_restaurants_selection.py
│   ├── cleared_reviews.csv               # Output from 3_clear_reviews.py
│   ├── cleared_reviews_removed_line_breaks.csv # Output from 3_clear_reviews.py with removed line breaks
│   └── assigned_restaurants.txt          # Output from 4_assign_annotator.py
│
├── scripts/                              # Scripts for data processing
│   ├── 1_get_restaurants.py              # Script to extract restaurant data
│   ├── 2_restaurants_selection.py        # Script for random selection
│   ├── 3_clear_reviews.py                # Script to clean review data
│   └── 4_assign_annotator.py             # Script to assign an annotator person
│
├── reports/                              # Reports and documentation
│   └── 1_report.md                       # First report detailing project progress
│
├── requirements.txt                      # Dependencies
│
├── README.md                             # Project overview and setup instructions
│
├── LICENSE.txt                           # Apache License 2.0 for the project
│
└── .gitignore                            # Files/folders to ignore in version control
```

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

### Key Terms:
- **Permissions**: This license allows for the use, modification, and distribution of the code, provided that all copies or substantial portions of the code include the original license.
- **Limitations**: The code is provided "as is," without warranty of any kind, express or implied, and without liability for any claims or damages arising from its use.
- **Attribution**: If you modify and distribute this code, you must include a prominent notice stating that you modified the files.

For more details, please refer to the full license text.

## Credits

### Team Members
- **[Nives Hüll](https://hulln.github.io/)** (nh23084@student.uni-lj.si, nhull@m.ffzg.hr): Role/Contribution (e.g., Project Coordinator, Data Acquisition)
- **[Ela Novak](https://github.com/ElaNovak4)**: (en87392@student.uni-lj.si, enovak3@ffzg.hr, ela.novak8@gmail.com): Role/Contribution (e.g., Data Analyst, Review Processing)
- **[Arja Hojnik](https://github.com/arjica)**(ah39150@student.uni-lj.si, ahojnik@ffzg.si): Role/Contribution (e.g., Programmer, Script Development) *(Please add your name and role here)*
- **[Meta Kokalj](https://github.com/meta899)** (mk10260@student.uni-lj.si; m.marguerite.k@gmail.com): Role/Contribution (e.g., Researcher, Documentation)
- **[Team Member 4]**: Role/Contribution (e.g., Reviewer, Quality Assurance) *(Please add your name and role here)*

### Acknowledgments
- Special thanks to our professor [Gaurish Thakkar](https://github.com/thak123/) for guidance and support throughout the project.

### Resources
- **Libraries**: 
  - [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
  - [Pandas](https://pandas.pydata.org/)
- **References**: (*Add any articles, papers, or online resources you referred to during the project*)

### Tools
- [**GitHub**](https://github.com/) for version control and collaboration.
- [**Python**](https://www.python.org/) for data processing and analysis.
- [**TripAdvisor® Review Scraper**](https://chromewebstore.google.com/detail/tripadvisor%C2%AE-review-scrap/pkbfojcocjkdhlcicpanllbeokhajlme) for review data extraction.
- [**ChatGPT**](https://chat.openai.com/) for assistance with coding, debugging, and support with other project-related tasks.
- (*Add any additional tools*)
