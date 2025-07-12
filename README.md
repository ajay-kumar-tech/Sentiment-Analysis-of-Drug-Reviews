# Sentiment Analysis of Drug Reviews

A pharmaceutical company has collected comments for various drug products by scraping from multiple online sources. The objective of this project is to build a sentiment analysis engine that can classify each review as **positive**, **negative**, or **neutral**, helping the company understand public perception.

## 📌 Project Objective

To develop a text classification model that can automatically detect sentiment from drug-related user reviews using Natural Language Processing and Machine Learning techniques.

---

## 📊 Data Description

### `train.csv`

This dataset contains labeled reviews.

| Variable  | Definition                                           |
|-----------|------------------------------------------------------|
| id        | Unique identifier for each record                   |
| comment   | Text pertaining to the drug product                 |
| product   | Drug name for which the review is provided          |
| sentiment | Target variable — 0 (positive), 1 (negative), 2 (neutral) |

### `test.csv`

Contains unlabeled data where the model is expected to predict the sentiment.

---

## 🔍 Exploratory Data Analysis (EDA)

- Performed EDA before and after text preprocessing
- Analyzed sentiment distribution and review lengths
- Checked for missing values and class imbalance
- Visualized frequent words using word clouds and bar plots

---

## ⚙️ Preprocessing Steps

- Text lowercasing
- Removal of punctuation and special characters
- Tokenization and stopword removal
- Feature engineering using **TF-IDF Vectorization**

---

## 🤖 Model Building

- Applied **SGDClassifier** (Stochastic Gradient Descent) for text classification
- Used stratified train-test split to preserve class distribution
- Hyperparameter tuning performed to improve performance

---

## ✅ Performance

- Achieved **86.2% accuracy** on the test data
- Efficiently classified reviews into **positive**, **negative**, and **neutral**

---

## 🧠 Tools & Technologies Used

- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, TF-IDF, SGDClassifier
- Jupyter Notebook

---

## 📈 Outcome

The model successfully classifies sentiment in pharmaceutical reviews and can be used by companies to:
- Monitor product perception
- Improve customer feedback analysis
- Drive data-driven decisions in product marketing

---

## 🚀 Future Work

- Experiment with deep learning models (LSTM, BERT)
- Deploy the model using Flask for real-time predictions
- Improve handling of multi-product reviews in a single comment

