import pandas as pd
import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# -------------------------------
# 1) Load data
# -------------------------------
df = pd.read_parquet("hf://datasets/nhull/tripadvisor-split-dataset/data/train.parquet")
print("Data shape:", df.shape)

# -------------------------------
# 2) Preprocess text exactly like training
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    return text

df["cleaned_review"] = df["review"].apply(preprocess_text)

# -------------------------------
# 3) Convert text -> sequences -> pad
#    The SAME tokenizer & same max_len used during training
# -------------------------------
with open("my_tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)
max_len = 200

X_eval_seq = tokenizer.texts_to_sequences(df["cleaned_review"])
X_eval_padded = pad_sequences(X_eval_seq, maxlen=max_len)

# Shifting labels from [1..5] to [0..4] (same as in training)
y_eval = df["label"] - 1
y_eval = y_eval.to_numpy()

# -------------------------------
# 4) Load final trained model
# -------------------------------
model = load_model("LSTM_model.h5")

# -------------------------------
# 5) Evaluate
# -------------------------------
loss, acc = model.evaluate(X_eval_padded, y_eval, batch_size=32)
print("Evaluation Loss:", loss)
print("Evaluation Accuracy:", acc)


# -------------------------------
# 6) Confussion matrix
# -------------------------------
from sklearn.metrics import confusion_matrix, classification_report

preds = model.predict(X_eval_padded)        # shape: (num_samples, 5)
y_pred = np.argmax(preds, axis=1)    # get predicted class indices
cm = confusion_matrix(y_eval, y_pred)

print("Confusion Matrix:\n", cm)

report = classification_report(y_eval, y_pred)
print("Classification Report:\n", report)
