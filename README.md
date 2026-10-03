# Credit Card Fraud Detection

A machine learning project that detects potentially fraudulent credit card transactions using Logistic Regression.

## Overview

Credit card fraud detection is a classification problem where fraudulent transactions are much less common than legitimate transactions. This project explores the dataset, handles the class imbalance, trains a Logistic Regression model, evaluates its performance, and deploys the trained model using Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Workflow

1. Loaded and explored the credit card transaction dataset.
2. Analyzed the distribution of legitimate and fraudulent transactions.
3. Identified severe class imbalance in the dataset.
4. Used undersampling during exploratory model development.
5. Split the data into training and testing sets.
6. Standardized the numerical features using `StandardScaler`.
7. Trained a Logistic Regression model.
8. Evaluated the model using accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC.
9. Saved the trained model and scaler using Joblib.
10. Built a Streamlit application to demonstrate predictions.

## Model

The project uses Logistic Regression for binary classification:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

Class weighting was used during the final model training to account for the strong class imbalance in the original dataset.

## Results

The final model achieved:

- Accuracy: 97.55%
- Fraud Recall: 92.00%
- ROC-AUC: 97.21%
- PR-AUC: 71.90%

Because fraud cases are rare, accuracy alone is not sufficient for evaluating the model. Precision, recall, F1-score and PR-AUC were also considered.

## Application

The trained model is integrated into a Streamlit application that allows users to test representative legitimate and fraudulent transactions.

The application displays:

- Model prediction
- Fraud probability
- Model performance metrics

## How to Run

Clone the repository:

```bash
git clone https://github.com/thatgirlmo1080/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Project Structure

```text
credit-card-fraud-detection/
│
├── app.py
├── fraud_detection.ipynb
├── fraud_detection_model.pkl
├── scaler.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project uses a credit card transactions dataset containing anonymized transaction features (V1–V28), transaction time, transaction amount, and a binary fraud label.

The original dataset is highly imbalanced, with fraudulent transactions representing only a very small proportion of the total transactions. The dataset itself is not included in this repository.

## Future Improvements

* Experiment with additional machine learning algorithms.
* Improve the Streamlit interface.
* Explore additional techniques for handling class imbalance.
* Tune the classification threshold based on the desired precision-recall trade-off.
