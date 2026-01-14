# Resume Analyzer – Text Classification System

## Project Overview
This project is a **Text Classification System** developed as part of the **AI/ML Engineer Intern Technical Assignment**.  
The system analyzes resume text and predicts the most relevant **job category** using machine learning techniques.

---

## Objective
To build a machine learning pipeline that:
- Takes raw resume text as input
- Performs text preprocessing
- Converts text into numerical features
- Trains classification models
- Evaluates performance using standard metrics
- Predicts the appropriate resume category

---

## Dataset
- **Source:** Resume Dataset
- **Format:** CSV file
- **Columns:**
  - `Resume`: Raw resume text
  - `Category`: Job category label

---

## Text Preprocessing
The following preprocessing steps were applied:
- Conversion to lowercase
- Removal of special characters and punctuation
- Tokenization
- Stopwords removal

---

## Feature Extraction
- **Technique:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **Max Features:** 3000
- TF-IDF was chosen to efficiently represent text data while reducing the impact of common words.

---

## Models Used
Two machine learning models were trained and compared:

### 1. Logistic Regression
- Performs well on high-dimensional sparse data
- Suitable for text classification tasks

### 2. Naive Bayes (MultinomialNB)
- Fast and effective for text-based problems
- Assumes feature independence

---

## Model Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

### Observations
- Logistic Regression achieved higher overall accuracy and F1-score compared to Naive Bayes.
- Logistic Regression handles sparse TF-IDF features more effectively, making it more suitable for resume classification.

The best-performing model was selected for deployment.

---

## Visualization
- A **confusion matrix** was plotted to visualize the classification performance of Logistic Regression.

---

## User Interface
- A simple **Streamlit web application** was built
- Users can paste resume text and get predicted job categories instantly

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit

---

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Train the model
```bash
python train.py
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
