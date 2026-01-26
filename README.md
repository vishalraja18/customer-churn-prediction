# Customer Churn Prediction Using Machine Learning

## Overview
Customer churn occurs when customers stop using a company’s services.  
This project predicts customer churn using machine learning to help businesses take proactive retention actions.

## Dataset
- Telco Customer Churn Dataset (Kaggle)
- 7,043 customers with demographic, usage, and billing information

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## Workflow
1. Data Cleaning & EDA
2. Feature Engineering & Leakage Removal
3. Model Training (Logistic Regression, Random Forest)
4. Model Evaluation (Confusion Matrix, ROC-AUC)
5. Business Insights & Feature Importance

## Results
- Logistic Regression Accuracy: 96%
- Recall (Churn): 90%
- ROC-AUC: 0.99

## Key Business Insights
- Month-to-month contracts have higher churn
- Higher satisfaction scores reduce churn
- Long-term contracts and bundled services improve retention

## Final Model
Logistic Regression was selected due to higher recall and better churn detection.

## Future Enhancements
- Streamlit dashboard
- Real-time churn prediction API
- Model retraining pipeline
