# Sentiment Analysis of TripAdvisor Reviews

## Overview
This project explores sentiment analysis, focusing on Michelin-starred restaurants and TripAdvisor reviews. It was carried out in two phases:

* **Phase 1: Creation and Annotation of a Sustainable Michelin Restaurants Dataset**  
   We curated a dataset of 125 reviews from Michelin-starred sustainable restaurants and annotated them using custom annotation guidelines.  
   **Dataset link**: [Sustainable Michelin Restaurants Dataset](https://docs.google.com/spreadsheets/d/1IJvYcopKYiTUHOE2AXrO-E-h3CX2t2HU-7HHh2p16cY/edit?usp=drive_link)

* **Phase 2: Model Training on a Larger Preexisting Dataset**  
   Due to the small size of the initial dataset, we used a larger, preexisting dataset of TripAdvisor hotel reviews for training. This dataset was modified and balanced to fit the project's requirements. Seven different NLP models, including Logistic Regression, LSTM, GRU, DistilBERT, TinyBERT, and RoBERTa were trained and evaluated for their performance on sentiment analysis.  
   **Dataset link**: [TripAdvisor Hotel Reviews Dataset](https://huggingface.co/datasets/nhull/tripadvisor-split-dataset-v2)

**Demo for All Models**: [Model Performance Demo](https://your-demo-link.com)  
**HuggingFace Repository for All Resources**: [HuggingFace Repository](https://huggingface.co/nhull/tripadvisor-project)

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Results and Discussion](#results-and-discussion)
- [License](#license)
- [Credits](#credits)

## Getting Started

### Prerequisites
You can install all the libraries used throughout the project by using the `requirements.txt` file  included in the repository.

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

## Repository Structure

```bash
TripAdvisor-Sentiment
│
├── data/                                 # General data files (1st phase)
│
├── dataset/                              # Datasets for training and evaluation
│
├── logs/                                 # Logs generated during experiments
│
├── models/                               # Trained models and model-related files
│
├── notes/                                # Miscellaneous notes (1st phase)
│
├── docs/                                 # Reports and documentation
│
├── results/                              # Results from model training and evaluation
│
├── scripts/                              # Scripts for data processing and modeling
│   ├── 1_annotation/                     # Annotation scripts
│   ├── 2_split_dataset/                  # Dataset splitting scripts
│   ├── 3_train_models/                   # Model training scripts
│   ├── 4_validate_models/                # Model validation scripts
│   └── 5_demo/                           # Demo-related scripts
│
│
├── CODE_OF_CONDUCT.md                    # Code of Conduct for contributors
│
├── LICENSE.txt                           # Apache License 2.0 for the project
│
├── README.md                             # Project overview
│
└── requirements.txt                      # Dependencies for the project
```

## Results and Discussion
### Models' Performance Overview

| Metric     | Logistic Regression |  LSTM  |  GRU   | BiLSTM | TinyBERT | RoBERTa | DistilBERT |
|------------|---------------------|--------|--------|--------|----------|---------|------------| 
| Accuracy   | 0.61                | 0.60   | 0.62   | 0.62   | 0.65     | 0.67    | 0.64       |
| Precision  | 0.61                | 0.60   | 0.62   | 0.62   | 0.64     | 0.67    | 0.64       |
| Recall     | 0.61                | 0.60   | 0.62   | 0.62   | 0.64     | 0.67    | 0.64       |
| F1-Score   | 0.61                | 0.60   | 0.62   | 0.62   | 0.64     | 0.67    | 0.64       |

**Note:** Additional metrics can be found in the final report, located in the `docs` folder of the repository.

1. **Logistic Regression**  
   Logistic Regression achieved an accuracy of **61.05%**, performing well for extreme sentiment labels (1 and 5) but struggled with mid-range ratings. This highlights the limitations of simpler methods in capturing nuanced sentiment distinctions.

2. **Deep Learning Models**  
   GRU, LSTM, and BiLSTM showed moderate improvements, with GRU achieving the best accuracy (**62.16%**) on the TripAdvisor dataset. These models demonstrated better generalization compared to Logistic Regression, though challenges like inter-class confusion persisted.

3. **Transformer Models**  
   Transformer-based models such as TinyBERT, DistilBERT, and RoBERTa exhibited the highest performance. TinyBERT achieved **65.35% accuracy**, benefiting from optimized training strategies. DistilBERT performed well for extreme sentiment labels but struggled with mid-range ratings, similar to other models. RoBERTa excelled with **67.22% test accuracy** on hotel reviews and showed strong generalization to other datasets, such as restaurant reviews.

### Dataset Limitations

The dataset used for training and evaluation posed significant challenges for achieving higher performance. Despite being balanced across sentiment labels, its relatively small size and inherent subjectivity in labeling impacted the models' ability to capture nuanced sentiment patterns. Subjective interpretations of reviews, particularly for mid-range ratings, likely contributed to misclassifications and variability in model performance.

### Summary

While transformer models outperformed traditional and deep learning approaches, improving mid-range sentiment classification remains a common challenge across all models. Addressing dataset limitations—such as increasing its size, refining annotation quality, and reducing subjectivity—would be crucial for achieving more robust results.


## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

### Key Terms:
- **Permissions**: This license allows for the use, modification, and distribution of the code, provided that all copies or substantial portions of the code include the original license.
- **Limitations**: The code is provided "as is," without warranty of any kind, express or implied, and without liability for any claims or damages arising from its use.
- **Attribution**: If you modify and distribute this code, you must include a prominent notice stating that you modified the files.

For more details, please refer to the full license text.

## Credits

### Team Members
- **[Nives Hüll](https://hulln.github.io/)** (nh23084@student.uni-lj.si, nhull@m.ffzg.hr)
- **[Ela Novak](https://github.com/ElaNovak4)** (en87392@student.uni-lj.si, enovak3@ffzg.hr, ela.novak8@gmail.com)
- **[Arja Hojnik](https://github.com/arjica)** (ah39150@student.uni-lj.si, ahojnik@ffzg.si)
- **[Meta Kokalj](https://github.com/meta899)** (mk10260@student.uni-lj.si, mkokalj@m.ffzg.hr, m.marguerite.k@gmail.com)
- **[Katarina Plementaš](https://github.com/KatarinaPlementas)** (kp88260@student.uni-lj.si, kplementa@m.ffzg.hr)
### Acknowledgments
- Special thanks to our professor [Gaurish Thakkar](https://github.com/thak123/) for guidance and support throughout the project.
- ChatGPT was used to assist with coding, debugging, and support with other project-related tasks as well as language improvement.

### References

- Aliyu, Yusuf et al. 2024. *‘Sentiment Analysis in Low-Resource Settings: A Comprehensive Review of Approaches, Languages, and Data Sources’.* IEEE Access, 12, 66883–66909. doi: [10.1109/ACCESS.2024.3398635](https://doi.org/10.1109/ACCESS.2024.3398635).
- Bharadwaj, Lakshay. 2023. *‘Sentiment Analysis in Online Product Reviews: Mining Customer Opinions for Sentiment Classification’.* International Journal For Multidisciplinary Research 5 (September). doi: [10.36948/ijfmr.2023.v05i05.6090](https://doi.org/10.36948/ijfmr.2023.v05i05.6090).
- Chifu, Adrian-Gabriel, and Sébastien Fournier. 2023. *‘Sentiment Difficulty in Aspect-Based Sentiment Analysis’.* Mathematics 11 (November):4647. doi: [10.3390/math11224647](https://doi.org/10.3390/math11224647).
- Ganie, Aadil Gani. 2023. *‘Presence of informal language, such as emoticons, hashtags, and slang, impact the performance of sentiment analysis models on social media text?’.* arXiv preprint arXiv:2301.12303. [https://arxiv.org/abs/2301.12303](https://arxiv.org/abs/2301.12303).
- Junichiro, Niimi. 2024. *‘Hotel Review Dataset (English)’.* [https://github.com/jniimi/tripadvisor_dataset](https://github.com/jniimi/tripadvisor_dataset).
- Michelin Guide. 2023. *‘Michelin Guide: Official Website’.* [https://guide.michelin.com](https://guide.michelin.com).
- Rangarjan, Prasanna, Bharathi Mohan Gurusamy, Gayathri Muthurasu, Rithani Mohan, Gundala Pallavi, Sulochana Vijayakumar, and Ali Altalbe. 2024. *‘The Social Media Sentiment Analysis Framework: Deep Learning for Sentiment Analysis on Social Media’.* International Journal of Electrical and Computer Engineering (IJECE) 14 (June):3394. doi: [10.11591/ijece.v14i3.pp3394-3405](https://doi.org/10.11591/ijece.v14i3.pp3394-3405).
- Sharma, Neeraj, A B M Shawkat Ali, and Ashad Kabir. 2024. *‘A Review of Sentiment Analysis: Tasks, Applications, and Deep Learning Techniques’.* International Journal of Data Science and Analytics, July, 1–38. doi: [10.1007/s41060-024-00594-x](https://doi.org/10.1007/s41060-024-00594-x).
- Tan, Kian Long, Chin Poo Lee, and Kian Ming Lim. 2023. *‘A Survey of Sentiment Analysis: Approaches, Datasets, and Future Research’.* Applied Sciences 13 (7). doi: [10.3390/app13074550](https://doi.org/10.3390/app13074550).
- TripAdvisor® Review Scraper. 2023. *‘TripAdvisor® Review Scraper Chrome Extension’.* [https://chromewebstore.google.com/detail/TripAdvisor%C2%AE%20Review%20Scraper/pkbfojcocjkdhlcicpanllbeokhajlme](https://chromewebstore.google.com/detail/TripAdvisor%C2%AE%20Review%20Scraper/pkbfojcocjkdhlcicpanllbeokhajlme).
