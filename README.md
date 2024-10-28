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
  - [Cloning the Repository](#cloning-the-repository)
  - [Running the Scripts](#running-the-scripts)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Credits](#credits)

## Getting Started

### Cloning the Repository
To clone this repository, run the following command:

```bash
git clone https://github.com/FFZG-NLP-2024/TripAdvisor-Sentiment.git
```

### Running the Scripts

The project includes two main scripts for data acquisition:

    1_get_restaurants.py: This script extracts restaurant data from HTML files stored in the data/src/ directory and saves it to combined_restaurants.txt.
    2_restaurants_selection.py: This script selects a balanced random sample of 50 restaurants from the combined list and saves them to selected_restaurants.txt.

To run the scripts, navigate to the project directory in your terminal and execute the following commands:

```bash
scripts/1_get_restaurants.py
scripts/2_restaurants_selection.py
```

## Repository Structure

```bash
TripAdvisor-Sentiment
│
├── data/                     # Raw data files
│   ├── src/                  # Source folder for HTML files
│   ├── combined_restaurants.txt # Output from 1_get_restaurants.py
│   └── selected_restaurants.txt  # Output from 2_restaurants_selection.py
│
├── scripts/                  # Scripts for data processing
│   ├── 1_get_restaurants.py  # Script to extract restaurant data
│   └── 2_restaurants_selection.py # Script for random selection
│
├── reports/                  # Reports and documentation
│   └── midterm_report.md      # Midterm report detailing project progress
│
├── requirements.txt          # Dependencies (if any)
│
├── README.md                 # Project overview and setup instructions
│
└── .gitignore                # Files/folders to ignore in version control
```

## License

To be added.

## Credits

To be added.
