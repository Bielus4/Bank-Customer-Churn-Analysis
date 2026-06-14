# 🏦 Bank-Customer-Churn-Analysis

> *"Acquiring a new customer can cost 5 to 25 times more than retaining an existing one."*  
> — Harvard Business Review

A machine learning project to identify bank customers at risk of churning, built on a classification pipeline combining exploratory data analysis with multiple supervised learning models.

---

## Overview

Customer churn — when a customer stops doing business with a company — is one of the most costly challenges in the banking industry. This project tackles that problem by building a predictive classification model based on features such as:

- Account balance
- Customer tenure
- Estimated salary
- Geography, age, gender, and more

The model enables financial institutions to proactively identify at-risk customers and deploy targeted retention strategies before they leave.

> ⚠️ **Note:** The dataset does not reflect real-world bank data. It is a hypothetical dataset designed for modeling purposes, as real banking data is highly sensitive.

---

## Dataset

**Source:** [Kaggle — Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset)

---

### Notebooks

| Notebook | Purpose |
|---|---|
| `banking_eda.ipynb` | Exploratory Data Analysis: statistics, feature distributions, correlation with churn target, and a reusable data cleaning function |
| `banking_modeling.ipynb` | Model training and evaluation: K-Nearest Neighbors, Logistic Regression, Random Forest — with hyperparameter tuning via Grid Search |

---

## Results

The best-performing model is a **Random Forest** classifier, optimized using Grid Search with the **F1-score** as the primary metric. F1-score was chosen to balance precision and recall — a practical choice given the class imbalance between retained and churned customers.

> 💡 Depending on the bank's strategy, an alternative objective may be preferable: maximizing **recall** to catch as many churners as possible, or maximizing **precision** to reduce false alarms in retention campaigns.

### Classification Report — Best Model (`rf_after_grid_search.pkl`)

```
              precision    recall  f1-score   support

           0       0.92      0.88      0.90       809
           1       0.58      0.68      0.62       191

    accuracy                           0.84      1000
   macro avg       0.75      0.78      0.76      1000
weighted avg       0.85      0.84      0.85      1000
```

### Interpretation

| Metric | Value | Meaning |
|---|---|---|
| **Overall Accuracy** | 84% | Correct predictions across both classes |
| **Churn Precision** | 58% | When the model predicts churn, it's right 58% of the time |
| **Churn Recall** | 68% | The model correctly identifies 68% of actual churners |
| **Churn F1-Score** | 0.62 | Balanced measure of precision and recall for the churn class |
