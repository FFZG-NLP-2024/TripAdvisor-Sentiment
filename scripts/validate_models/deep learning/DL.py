import os
import re
import numpy as np
import pandas as pd
import keras_tuner as kt
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load the dataset
from datasets import load_dataset
dataset = load_dataset('nhull/tripadvisor-split-dataset-v2')

train_data = pd.DataFrame(dataset['train'])
valid_data = pd.DataFrame(dataset['validation'])
test_data = pd.DataFrame(dataset['test'])

# Step 2: Preprocessing - Text cleaning
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    return text

train_reviews = [preprocess_text(r) for r in train_data['review']]
valid_reviews = [preprocess_text(r) for r in valid_data['review']]
test_reviews  = [preprocess_text(r) for r in test_data['review']]

# Shifting labels from [1..5] to [0..4]
y_train = np.array(train_data['label']) - 1
y_valid = np.array(valid_data['label']) - 1
y_test  = np.array(test_data['label']) - 1

# Step 3: Tokenizer setup
max_words = 10000
max_len   = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_reviews)

# Convert text -> sequences
X_train = tokenizer.texts_to_sequences(train_reviews)
X_valid = tokenizer.texts_to_sequences(valid_reviews)
X_test  = tokenizer.texts_to_sequences(test_reviews)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_len)
X_valid = pad_sequences(X_valid, maxlen=max_len)
X_test  = pad_sequences(X_test,  maxlen=max_len)

# Save tokenizer
save_dir = 'models/deep_learning/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(save_dir, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved as 'models/deep_learning/tokenizer.pkl'")

# Step 4: Load GloVe and create embedding matrix
glove_path = "glove.6B.100d.txt"
embedding_dim = 100

# Build embedding index
embedding_index = {}
with open(glove_path, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coeffs

# Build embedding matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Step 5: Define multiple model-building functions (LSTM, GRU, BiLSTM)
def build_lstm_model(hp):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=max_words,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=False
        )
    )
    model.add(
        LSTM(
            units=hp.Int("units", min_value=64, max_value=256, step=64),
            return_sequences=False
        )
    )
    model.add(
        Dropout(hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1))
    )
    model.add(
        Dense(
            units=hp.Int("dense_units", min_value=32, max_value=128, step=32),
            activation="relu"
        )
    )
    model.add(Dense(5, activation="softmax"))

    model.compile(
        optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_gru_model(hp):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=max_words,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=False
        )
    )
    model.add(
        GRU(
            units=hp.Int("units", min_value=64, max_value=256, step=64),
            return_sequences=False
        )
    )
    model.add(
        Dropout(hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1))
    )
    model.add(
        Dense(
            units=hp.Int("dense_units", min_value=32, max_value=128, step=32),
            activation="relu"
        )
    )
    model.add(Dense(5, activation="softmax"))

    model.compile(
        optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_bilstm_model(hp):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=max_words,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=False
        )
    )
    model.add(
        Bidirectional(
            LSTM(
                units=hp.Int("units", min_value=64, max_value=256, step=64),
                return_sequences=False
            )
        )
    )
    model.add(
        Dropout(hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1))
    )
    model.add(
        Dense(
            units=hp.Int("dense_units", min_value=32, max_value=128, step=32),
            activation="relu"
        )
    )
    model.add(Dense(5, activation="softmax"))

    model.compile(
        optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Step 6: Tuning & evaluation function
def tune_and_evaluate(build_fn, project_name):
    tuner = kt.Hyperband(
        build_fn,
        objective="val_accuracy",
        max_epochs=5,  # quick search
        factor=3,
        directory="hyperparam_tuning",
        project_name=project_name
    )

    stop_early = EarlyStopping(monitor='val_loss', patience=2)

    # Tuner search
    tuner.search(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=5,
        batch_size=32,
        callbacks=[stop_early]
    )

    # Retrieve best HPs
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nBest HP for {project_name}: {best_hp.values}")

    # Build best model
    model = tuner.hypermodel.build(best_hp)

    # Retrain for more epochs
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=10,
        batch_size=32,
        callbacks=[stop_early]
    )

    # Evaluate on training data
    train_preds = model.predict(X_train)
    y_train_pred = np.argmax(train_preds, axis=1)

    # Evaluate on validation data
    val_preds = model.predict(X_valid)
    y_val_pred = np.argmax(val_preds, axis=1)

    # Adjust labels and predictions from 0-4 to 1-5
    y_train_adj = y_train + 1
    y_train_pred_adj = y_train_pred + 1
    y_val_adj = y_valid + 1
    y_val_pred_adj = y_val_pred + 1

    # Step 7: Save CSV files for training and validation results
    def save_results_to_csv(reviews, true_labels, pred_labels, filename):
        data = {
            "Review": reviews,
            "True Label": true_labels,
            "Predicted Label": pred_labels,
            "Difference": [pred - true for pred, true in zip(pred_labels, true_labels)],
            "Absolute Difference": [abs(pred - true) for pred, true in zip(pred_labels, true_labels)]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Saved results to {filename}")

    save_results_to_csv(train_reviews, y_train_adj, y_train_pred_adj, f"training_results_{project_name}.csv")
    save_results_to_csv(valid_reviews, y_val_adj, y_val_pred_adj, f"validation_results_{project_name}.csv")

    # Step 8: Save confusion matrix plots for training and validation
    def save_confusion_matrix(true_labels, pred_labels, title, filename):
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 6), yticklabels=range(1, 6))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(title)
        plt.savefig(filename)
        plt.show()
        print(f"Saved confusion matrix to {filename}")

    save_confusion_matrix(y_train_adj, y_train_pred_adj, f'Confusion Matrix (Training) - {project_name}', f"confusion_matrix_train_{project_name}.png")
    save_confusion_matrix(y_val_adj, y_val_pred_adj, f'Confusion Matrix (Validation) - {project_name}', f"confusion_matrix_val_{project_name}.png")

    # Step 9: Print classification reports
    print("\nTraining Classification Report:")
    print(classification_report(y_train_adj, y_train_pred_adj))

    print("\nValidation Classification Report:")
    print(classification_report(y_val_adj, y_val_pred_adj))

    # Save the model
    model.save(f"best_{project_name}_model.h5")
    print(f"Saved {project_name} model as 'best_{project_name}_model.h5'")

    return history.history['val_accuracy'][-1]

# Step 10: Tune each model
lstm_acc   = tune_and_evaluate(build_lstm_model,   "LSTM_tuning")
gru_acc    = tune_and_evaluate(build_gru_model,    "GRU_tuning")
bilstm_acc = tune_and_evaluate(build_bilstm_model, "BiLSTM_tuning")

print("\n==== Final Results ====")
print("LSTM   Validation Accuracy:",   lstm_acc)
print("GRU    Validation Accuracy:",   gru_acc)
print("BiLSTM Validation Accuracy:",   bilstm_acc)