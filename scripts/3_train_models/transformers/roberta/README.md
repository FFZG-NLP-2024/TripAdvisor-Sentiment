# Sentiment Analysis Model: Predicting Customer Satisfaction Ratings

## Objective
The primary aim of this research was to develop and evaluate a sentiment analysis model capable of predicting customer satisfaction ratings (on a 1-5 scale) for hotel reviews, with additional testing on restaurant reviews to explore cross-domain capabilities.

---

## Methodology

### Model Architecture
- **Base Model:** RoBERTa
- **Task:** Multi-class classification (5 rating classes)

### Training Strategy
- **Datasets:**
  - **Hotel Reviews:** TripAdvisor hotel reviews
  - **Restaurant Reviews:** 125 TripAdvisor restaurant reviews annotated by the project group
- **Evaluation Strategy:** Train/validation/test split for hotel reviews; cross-domain testing on restaurant reviews

### Implementation Details
- **Preprocessing:** Text tokenization with a max length of 128
- **Hyperparameters:**
  - Batch size: 16 (hotels), 8 (restaurants)
  - Learning rate: 5e-5
  - Training epochs: 3
  - Optimizer: AdamW with weight decay of 0.01
  - Scheduler: Linear with warmup

---

## Performance Metrics

### Training and Validation Performance (Hotels)
- **Training Loss:** Decreased from 0.8893 (epoch 1) to 0.5313 (epoch 3), indicating effective learning
- **Validation Accuracy:** 65.94%
- **Validation Metrics:**
  - Consistent precision, recall, and F1-score across classes
  - Class 5 performed best

### Test Performance

#### Hotel Test Dataset
- **Accuracy:** 67.22%
- **Class Performance:**
  - Class 5 performed best

#### Restaurant Test Dataset (Cross-domain Evaluation)
- **Accuracy:** 74.40%
- **Performance Variability:**
  - Class 5 performed best
  - Significant variation across classes

### Detailed Metrics
#### Hotels
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

#### Restaurants
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

## Evaluation

### Metrics Used
- Classification reports with precision, recall, and F1-score
- Confusion matrices for error analysis
- Instance-level prediction analysis

### Cross-domain Testing
- **Dataset:** TripAdvisor restaurant reviews (125 annotated reviews)
- **Purpose:** Explore generalization capability to a related but distinct domain

---

## Research Significance
This study focuses on sentiment analysis for hotel reviews while exploring cross-domain applicability to restaurant reviews. It highlights:
- Differences between domains (e.g., broader hotel services vs. focused restaurant aspects)
- Potential for cross-domain sentiment analysis to inform business decisions
- Insights into generalization of sentiment analysis models across service domains

---

## Future Improvements

1. **Data Augmentation:**
   - Augment training data to prevent overfitting and ensure diverse learning.

2. **Domain-specific Fine-tuning:**
   - Fine-tune the model for specific domains to capture nuanced differences.

3. **Enhanced Evaluation Strategies:**
   - Introduce evaluation checkpoints during training to monitor stability.

4. **Hyperparameter Optimization:**
   - Experiment with different hyperparameter combinations for improved accuracy and consistency.

---
