# 🏦 Bank Customer Churn Prediction

A predictive modeling project aimed at identifying high-risk customers likely to leave the bank. The project combines business understanding with data science techniques to support retention strategy and maximize ROI, completed as part of a class assignment.

---

## 📌 Context
This project was conducted as part of a graduate-level analytics course. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction) and represents structured data on customer behavior, demographics, and banking engagement.

---

## 🎯 Objective
To develop and evaluate classification models that predict the likelihood of a customer churning, and design a profit-maximizing retention strategy based on churn probability thresholds and business cost-benefit metrics.

---

## 🛠 Languages & Libraries
- **Language**: R  
- **Libraries**: `randomForest`, `caret`, `glmnet`, `class`, `pROC`, `ggplot2`, `corrplot`, `GGally`, `dplyr`

---

## 🔍 Methods Applied
- Data cleaning & factor conversion
- Exploratory analysis (pair plots, boxplots, correlation)
- Feature selection and engineering (e.g., engagement score)
- **Modeling**:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Lasso Regression
- ROC/AUC evaluation and model comparison
- Profit curve optimization and ROI calculation

---

## 🧠 Modeling Summary
- **Best Model**: ✅ Random Forest (AUC = Highest among all models)
- **Logistic Regression**: Good baseline; key predictors included age, geography, and activity status
- **KNN**: 87.5% accuracy; good sensitivity but lower specificity for churn
- **Lasso**: Useful for feature selection and variable interpretation

---

## 💰 Business Impact
- Applied a cost-benefit matrix to determine optimal threshold for retention offers
- Threshold: 0.56 → Send offers to 15,486 customers
- Profit: $3,475,752  
- ROI: **224.5%**

---

## 📈 Results
- Random Forest provided the best predictive performance.
- Strategic deployment of personalized offers to high-risk customers can lead to a significant reduction in churn and increase in net profit.

---

## 👥 Team
Berry Tian, Arnav Bhasin, Nitya Reddy, Charlie Liu, Youran Yu  
Section B – Team 5
