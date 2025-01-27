# RoBERTA Sentiment Analysis Model: Performance Metrics


## Table of Contents
1. [Objective](#objective)
2. [Methodology](#methodology)
   - [Model Architecture](#model-architecture)
   - [Training Strategy](#training-strategy)
   - [Implementation Details](#implementation-details)
3. [Performance Metrics](#performance-metrics)
   - [Training and Validation Performance (Hotels)](#training-and-validation-performance-hotels)
   - [Test Performance](#test-performance)
     - [Hotel Test Dataset](#hotel-test-dataset)
     - [Restaurant Test Dataset (Cross-domain Evaluation)](#restaurant-test-dataset-cross-domain-evaluation)
   - [Detailed Metrics](#detailed-metrics)
     - [Hotels](#hotels)
     - [Restaurants](#restaurants)
4. [Evaluation](#evaluation)
   - [Metrics Used](#metrics-used)
   - [Cross-domain Testing](#cross-domain-testing)
5. [Research Significance](#research-significance)
6. [Future Improvements](#future-improvements)
7. [Analysis of Predictions](#analysis-of-predictions)
   - [Steps](#steps)
  
---

## 1. Objective
The primary aim of this research was to develop and evaluate a sentiment analysis model capable of predicting customer satisfaction ratings (on a 1-5 scale) for hotel reviews, with additional testing on restaurant reviews to explore cross-domain capabilities.

---

## 2. Methodology

### 2.1 Model Architecture
- **Base Model:** RoBERTa
- **Task:** Multi-class classification (5 rating classes)

### 2.2 Training Strategy
- **Datasets:**
  - **Hotel Reviews:** TripAdvisor hotel reviews
  - **Restaurant Reviews:** 125 TripAdvisor restaurant reviews annotated by the project group
- **Evaluation Strategy:** Train/validation/test split for hotel reviews; cross-domain testing on restaurant reviews

### 2.3 Implementation Details
- **Preprocessing:** Text tokenization with a max length of 128
- **Hyperparameters:**
  - Batch size: 16 (hotels), 8 (restaurants)
  - Learning rate: 5e-5
  - Training epochs: 3
  - Optimizer: AdamW with weight decay of 0.01
  - Scheduler: Linear with warmup

---

## 3. Performance Metrics

### 3.1 Training and Validation Performance (Hotels)
- **Training Loss:** Decreased from 0.8893 (epoch 1) to 0.5313 (epoch 3), indicating effective learning
- **Validation Accuracy:** 65.94%
- **Validation Metrics:**
  - Consistent precision, recall, and F1-score across classes
  - Class 5 performed best

### 3.2 Test Performance

#### 3.2.1 Hotel Test Dataset
- **Accuracy:** 67.22%
- **Class Performance:**
  - Class 5 performed best

#### 3.2.2 Restaurant Test Dataset (Cross-domain Evaluation)
- **Accuracy:** 74.40%
- **Performance Variability:**
  - Class 5 performed best
  - Significant variation across classes

### 3.3 Detailed Metrics
#### 3.3.1 Hotels
- **Overall Metrics:**
  
  - Accuracy: 0.672
  - Precision: 0.674
  - Recall: 0.672
  - F1 Score: 0.674
- **Class-wise Metrics:**
  
  | Label | Precision | Recall | F1-score | Support |
  |-------|-----------|--------|----------|---------|
  | 1     | 0.75      | 0.76   | 0.75     | 1600    |
  | 2     | 0.58      | 0.59   | 0.59     | 1600    |
  | 3     | 0.65      | 0.62   | 0.64     | 1600    |
  | 4     | 0.62      | 0.62   | 0.62     | 1600    |
  | 5     | 0.77      | 0.77   | 0.77     | 1600    |

#### 3.3.2 Restaurants
- **Overall Metrics:**
  
  - Accuracy: 0.744
  - Precision: 0.81
  - Recall: 0.74
  - F1 Score: 0.73
- **Class-wise Metrics:**
  
  | Label | Precision | Recall | F1-score | Support |
  |-------|-----------|--------|----------|---------|
  | 1     | 1.00      | 0.14   | 0.25     | 14      |
  | 2     | 0.00      | 0.00   | 0.00     | 6       |
  | 3     | 0.40      | 0.75   | 0.52     | 8       |
  | 4     | 0.79      | 0.52   | 0.63     | 21      |
  | 5     | 0.89      | 0.97   | 0.93     | 76      |

---

## 4. Evaluation

### 4.1 Metrics Used
- Classification reports with precision, recall, and F1-score
- Confusion matrices for error analysis
- Instance-level prediction analysis

### 4.2 Cross-domain Testing
- **Dataset:** TripAdvisor restaurant reviews (125 annotated reviews)
- **Purpose:** Explore generalization capability to a related but distinct domain

---

## 5. Research Significance
This study focuses on sentiment analysis for hotel reviews while exploring cross-domain applicability to restaurant reviews. It highlights:
- Differences between domains (e.g., broader hotel services vs. focused restaurant aspects)
- Potential for cross-domain sentiment analysis to inform business decisions
- Insights into generalization of sentiment analysis models across service domains

---

## 6. Future Improvements

1. **Data Augmentation:**
   - Augment training data to prevent overfitting and ensure diverse learning.

2. **Domain-specific Fine-tuning:**
   - Fine-tune the model for specific domains to capture nuanced differences.

3. **Enhanced Evaluation Strategies:**
   - Introduce evaluation checkpoints during training to monitor stability.

4. **Hyperparameter Optimization:**
   - Experiment with different hyperparameter combinations for improved accuracy and consistency.

---
# 7. Analysis Of Predictions

In this analysis, we processed the prediction results from the model on the test datasets for both **Hotels** and **Restaurants**. The goal was to extract only those instances where the absolute difference between the **True Label** and **Predicted Label** was significantly high, indicating potential misclassifications. This step helps us identify more challenging or ambiguous predictions for further analysis.

### 7.1 Steps:

1. **Mapping Predictions to the Original Dataset**:
   - We started by mapping the predictions from the model back to the original test datasets (for both Hotels and Restaurants). 
   - Each row from the prediction result CSV file was matched with the corresponding review text from the test dataset using the index. This allowed us to create a comprehensive file that included the original review text, true labels, predicted labels, and the absolute difference between true and predicted values.

2. **Filtering Predictions Based on Absolute Difference**:
   - After mapping, we filtered the rows based on the absolute difference between the true label and predicted label:
     - **For Hotels**: We extracted cases where the **absolute difference** between true and predicted labels was **3 or more**.
     - **For Restaurants**: We focused on cases where the **absolute difference** between true and predicted labels was **2 or more** (since there were only two instances with an absolute difference equal to 2 in the restaurant dataset).

3. **Filtered CSV Files**:
   - The final result was a filtered dataset for both Hotels and Restaurants, containing only those reviews where the model predictions were significantly off from the true labels. This can be used for further analysis of difficult or ambiguous predictions.

### Purpose of Filtering:
- The main purpose of this filtering step was to focus on the predictions where the model performed poorly or struggled with high misclassification errors (high absolute differences). This helps in understanding the weaknesses of the model and can guide further improvements or analysis.
