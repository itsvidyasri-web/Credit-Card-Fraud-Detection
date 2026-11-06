# Credit Card Fraud Detection Using Machine Learning

## Live Demo

**Try the Application:**  
https://credit-card-fraud-detection-8uiwy5gihdoeawva5sfukt.streamlit.app/

## Project Overview

Credit card fraud is a major challenge in digital financial transactions. This project uses Machine Learning to identify potentially fraudulent transactions and distinguish them from genuine transactions.

The application provides an interactive Streamlit dashboard where users can upload transaction data in CSV format and receive fraud predictions.

## Objectives

- Analyze credit card transaction data.
- Identify patterns associated with fraudulent transactions.
- Build a Machine Learning classification model.
- Evaluate model performance using classification metrics.
- Develop an interactive web application for fraud detection.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Logistic Regression
- StandardScaler
- Pickle

## Dataset

**Dataset:** Credit Card Fraud Detection

**Source:** Kaggle – Machine Learning Group (ULB)

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset details:
- Total transactions: 284,807
- Total columns: 31
- Input features: 30
- Target column: Class
- Class 0: Genuine transaction
- Class 1: Fraudulent transaction

The dataset contains anonymized transaction features, including V1 to V28, Time, and Amount.

## Machine Learning Approach

This project uses Logistic Regression for binary classification.

### Workflow

1. Load and explore the dataset.
2. Check missing values and class distribution.
3. Separate input features and target variable.
4. Split the dataset into training and testing sets.
5. Apply StandardScaler through a Machine Learning pipeline.
6. Train the Logistic Regression model.
7. Evaluate predictions using classification metrics.
8. Save the trained model using Pickle.
9. Integrate the model with Streamlit.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

These metrics help assess how effectively the model identifies fraudulent transactions.

## Application Features

- Interactive Streamlit dashboard.
- CSV transaction data upload.
- Fraudulent and genuine transaction classification.
- Fraud probability display.
- Transaction summary metrics.
- Downloadable prediction results.

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── main.py
├── requirements.txt
├── fraud_model.pkl
└── README.md
```

The dataset file `creditcard.csv` is used locally for model training and is not required for dashboard users to upload.

## How to Run Locally

1. Clone the repository:

```bash
https://github.com/itsvidyasri-web/Credit-Card-Fraud-Detection
```

2. Navigate to the project folder:

```bash
cd Credit-Card-Fraud-Detection
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

5. Open the local URL displayed in your terminal.

## Future Enhancements

- Explore advanced fraud detection algorithms.
- Improve fraud detection performance on imbalanced data.
- Add interactive transaction visualizations.
- Integrate secure real-time transaction data sources.
- Explore model monitoring and performance tracking.

## Disclaimer

This project is developed for educational and learning purposes using an anonymized historical dataset. It is a Machine Learning prototype and is not intended to replace real-world banking fraud prevention systems.

## Author

**Data Science Project**

Developed as part of my hands-on learning journey in Data Science and Machine Learning.

## Live Application

https://credit-card-fraud-detection-8uiwy5gihdoeawva5sfukt.streamlit.app/
