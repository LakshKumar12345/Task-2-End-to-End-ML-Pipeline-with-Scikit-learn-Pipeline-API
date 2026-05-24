# 📊 Telco Customer Churn Prediction  
### End-to-End Machine Learning Pipeline (Classification Project)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

---

## 🧠 Project Overview

This project focuses on predicting customer churn for a telecom company using machine learning. The goal is to identify customers who are likely to leave the service so that the company can take proactive retention actions.

We built a complete end-to-end ML pipeline including:
- Data cleaning and preprocessing
- Feature engineering
- Multiple classification models
- Hyperparameter tuning (GridSearchCV)
- Model evaluation using multiple metrics
- Business insight extraction

---

## 📂 Dataset Information

- **Dataset:** IBM Telco Customer Churn Dataset  
- **Target Variable:** `Churn`  
  - 1 → Customer left (Churned)  
  - 0 → Customer stayed  

### Key Features:
- Contract type
- Tenure
- Monthly charges
- Total charges
- Internet service
- Customer demographics

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Converted `TotalCharges` to numeric
- Handled missing values using median imputation
- Encoded categorical variables using OneHotEncoder
- Standardized numerical features using StandardScaler

---

### 2️⃣ Feature Engineering
- Separated numerical and categorical features
- Built preprocessing pipeline using ColumnTransformer

---

### 3️⃣ Models Used

| Model | Description |
|------|-------------|
| Logistic Regression | Simple baseline linear model |
| Random Forest | Ensemble non-linear model |
| Random Forest + GridSearchCV | Tuned model |

---

## 📊 Model Evaluation Metrics

- Accuracy
- Precision
- Recall (most important for churn detection)
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 📉 Results Summary

| Model | Recall (Churn) | ROC-AUC | Insight |
|------|---------------|--------|--------|
| Logistic Regression | ⭐ Best recall | High | Best at identifying churners |
| Random Forest (Baseline) | Lower | Good | Slightly overfits |
| Random Forest (Tuned) | No improvement | Good | Similar to baseline |

---

## ⚠️ Important Insight (GridSearchCV Result)

Even after hyperparameter tuning using GridSearchCV, the best parameters remained close to default Random Forest settings:

- n_estimators = 100  
- max_depth = None  
- min_samples_split = 2  
- min_samples_leaf = 1  

### 🔥 Conclusion:
- Random Forest was already near optimal
- Tuning did not significantly improve performance
- Surprisingly, Logistic Regression performed better in recall

👉 This shows that:
> Simple models can outperform complex models on structured tabular data.

---

## 📌 Business Insights

### 🔥 Top churn drivers:
- Month-to-month contracts
- Low tenure customers
- High monthly charges
- Early-stage customers (new users)

### 🎯 Business Impact:
- Focus retention campaigns on new customers
- Promote long-term contracts
- Offer targeted discounts for high-risk users
- Early intervention significantly reduces churn

---

## 📸 Visual Results

# 📸 Visual Results

### 🔵 ROC Curve Comparison
![ROC Curve](images/roc_curve.png)

---

### 🔷 Confusion Matrix - Logistic Regression
![Confusion Matrix Logistic Regression](images/confusion_matrix_lr.png)

---

### 🟢 Confusion Matrix - Random Forest
![Confusion Matrix Random Forest](images/confusion_matrix_rf.png)

---

### 🌲 Feature Importance Plot
![Feature Importance](images/feature_importance.png)

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 👨‍💻 Author & Future Improvements

### 👨‍💻 Author
Laksh Kumar 
Machine Learning Intern
Developers Hub Cooperation

### 🚀 Future Improvements
- Experiment with XGBoost / LightGBM for better performance
- Handle class imbalance using SMOTE
- Deploy model using Streamlit dashboard
- Add SHAP for model explainability
- Build Flask/FastAPI backend for production use
