import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import TrainingArguments, Trainer, DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
import re


# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# # Load standard English stopwords
# stop_words = set(stopwords.words('english'))


warnings.filterwarnings('ignore')


print(torch.backends.mps.is_available())  # Should return False
print(torch.backends.mps.is_built())
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
print(torch.backends.mps.is_available())  # Should return False
print(torch.backends.mps.is_built())


# Set device (Use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
def load_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    return train_df, test_df

train_df, test_df = load_data()
print(train_df.head())

# # Clean text function
# def clean_text(text):
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
#     text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
#     text = re.sub(r'[^A-Za-z0-9 ]+', '', text.lower())  # Remove special characters
#     text = re.sub(r"can't", "cannot", text)  # Expand contractions
#     text = re.sub(r"n't", " not", text)  # Expand negations
#     words = text.split()  # Tokenize text
#     words = [word for word in words if word not in stop_words]  # Remove stopwords
#     text = ' '.join(words)  # Reconstruct cleaned text
#     return text.strip()

# # Apply text cleaning
# train_df["text"] = train_df["text"].apply(clean_text)
# test_df["text"] = test_df["text"].apply(clean_text)

# Load DistilBERT tokenizer & model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

# Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    list(train_df["text"]), list(train_df["target"]), test_size=0.2, stratify=train_df["target"]
)

# Tokenize Data
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=256, return_tensors="pt")
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=256, return_tensors="pt")

# Move to Device
X_train_tokenized = {key: val.to(device) for key, val in X_train_tokenized.items()}
X_val_tokenized = {key: val.to(device) for key, val in X_val_tokenized.items()}

# Define Dataset Class
class DisasterTweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels).to(device)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create train & validation datasets
train_dataset = DisasterTweetDataset(X_train_tokenized, y_train)
val_dataset = DisasterTweetDataset(X_val_tokenized, y_val)

# Training Arguments (Optimized)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,  # Number of training epochs
    per_device_train_batch_size=32,  # Batch size per GPU/CPU
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    fp16=True,  # Mixed precision for faster training
    learning_rate=3e-5,  # Optimized learning rate
    weight_decay=0.01,  # Regularization
    warmup_ratio=0.06 # Helps stabilize training
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train Model
trainer.train()

# Evaluate Model
eval_results = trainer.evaluate()
print("\nTrainer Evaluation Results:", eval_results)



# --------------------
# Compute Accuracy, Precision, Recall, and F1-score on Validation Set
# --------------------

def compute_metrics():
    """Compute validation accuracy, precision, recall, and F1-score."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for text, label in zip(X_val, y_val):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits

        pred_label = torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()
        all_preds.append(pred_label)
        all_labels.append(label)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n**Validation Performance Metrics:**")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Run evaluation
compute_metrics()



# --------------------
# Retrain Model on 100% of the Data
# --------------------

print("\nRetraining model on 100 percent of the training data...")

# Combine training and validation data
full_texts = list(train_df["text"])  # Use full dataset
full_labels = list(train_df["target"])

# Tokenize full dataset
full_tokenized = tokenizer(full_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

# Move to Device
full_tokenized = {key: val.to(device) for key, val in full_tokenized.items()}

# Define New Dataset for Full Training
full_train_dataset = DisasterTweetDataset(full_tokenized, full_labels)

# New Trainer for Full Dataset Training
full_training_args = TrainingArguments(
    output_dir="./results_full",
    num_train_epochs=4,  # Use fewer epochs (model already trained before)
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    evaluation_strategy="no",  # No validation since using 100% of data
    save_strategy="epoch",
    logging_dir="./logs_full",
    save_total_limit=1,
    report_to="none",
    fp16=True,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
)

full_trainer = Trainer(
    model=model,
    args=full_training_args,
    train_dataset=full_train_dataset
)

# Retrain Model on Full Data
full_trainer.train()
print("\nModel retrained on full dataset.")



# --------------------
# Predict on Test Data
# --------------------

def predict_with_model(text):
    """Generate predictions on test data."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()

# Directory where submission files are stored
submission_dir = "submission"

# Get a list of existing submission files
existing_files = [f for f in os.listdir(submission_dir) if re.match(r"submission-\d+\.csv", f)]

# Extract existing numbers
existing_numbers = [int(re.search(r"submission-(\d+)\.csv", f).group(1)) for f in existing_files]

# Determine next available submission number
next_number = max(existing_numbers) + 1 if existing_numbers else 1

# Define new submission file path
final_csv_path = f"{submission_dir}/submission-{next_number}.csv"


# Predict and save to submission file
test_df["target"] = test_df["text"].apply(predict_with_model)
submission = test_df[['id', 'target']]
submission.to_csv(final_csv_path, index=False)
print(f"Final submission saved at: {final_csv_path}")

