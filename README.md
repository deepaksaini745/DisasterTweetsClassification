# Disaster Tweet Classification

This repository contains a machine learning pipeline to classify whether a tweet is related to a real disaster or not. The project was built using traditional NLP techniques and tested with multiple classifiers to determine the most effective approach.

---

## Project Overview

Social media platforms often provide real-time updates during natural disasters. The objective of this project is to automatically classify tweets as **disaster-related** or **not disaster-related**, enabling quicker filtering and response by emergency services or news organizations.

---

## Approach

1. **Text Preprocessing**

   * Lowercasing
   * Removing punctuation, URLs, and mentions
   * Tokenization
   * Stopword removal
   * Lemmatization

2. **Feature Extraction**

   * TF-IDF vectorization

3. **Model Training**

   * Logistic Regression
   * Random Forest
   * Multinomial Naive Bayes
   * XGBoost

4. **Evaluation**

   * F1 Score
   * Accuracy
   * Confusion Matrix

---

## Results

* The best model achieved an **F1 score of 0.83** on the test set.
* XGBoost and Naive Bayes performed best in terms of both speed and accuracy.
* Further improvements can be made with LSTM/BERT-based models (not part of this version).

---

## Files

* `Classifying Disaster Tweets.py`: Main script containing data preprocessing, training, evaluation, and model comparison logic.

---

## Future Work

* Integrate pre-trained transformers (e.g., BERT)
* Build a simple UI for classification
* Deploy via Flask or FastAPI for real-time inference

