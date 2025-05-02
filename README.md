# 🏦 Bank Customer Churn Prediction

A machine learning project aimed at predicting customer churn for a retail bank using classification models. We evaluated logistic regression, random forest, KNN, and lasso, and developed a cost-optimized targeting strategy for retention offers.

---

## 📂 Dataset
- Source: [Kaggle – Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)
- Records include customer demographics, financial metrics, and a churn label (`Exited`)

---

## 🎯 Problem Statement
Identify customers at risk of churn and design a targeted retention strategy using churn probability and a cost-benefit framework.

---

## 🛠 Tech Stack

- **Language**: R  
- **Key Libraries**: `caret`, `randomForest`, `glmnet`, `pROC`, `ggplot2`, `corrplot`, `GGally`, `dplyr`

---

## ⚙️ Methods

### Data Preparation
- Cleaned and encoded categorical variables
- Removed IDs and surnames
- Feature engineered an Engagement Score

### Models
- ✅ **Random Forest** (best performance, AUC = highest)
- Logistic Regression
- K-Nearest Neighbors (k=5)
- Lasso Regression with cross-validation

### Evaluation
- ROC/AUC for each model
- Feature importance plotted
- Profit curve and ROI calculated for different churn thresholds

---

## 📈 Key Results

| Model             | Evaluation Metric        | Note |
|------------------|--------------------------|------|
| **Random Forest** | AUC (best), 92.5% acc    | Final model |
| Logistic Reg.     | Interpretable baseline   | Age & activity most predictive |
| KNN               | 87.5% acc, 94% recall on non-churners | Moderate precision |
| Lasso             | Shrunk unimportant vars  | Used for feature selection |

---

## 💡 Business Strategy

- Applied a **cost-benefit matrix** to estimate profit from offering $100 retention incentives
- **Optimal threshold**: 0.56  
- **Targeted customers**: 15,486  
- **ROI**: 224.5%  
- **Profit**: $3.47M (simulated)

---

## 📁 Files

- `BankChurnTeam5.R`: Full pipeline – EDA, modeling, evaluation, deployment
- `Final Report.pdf`: Business interpretation and deliverables

---

## 👤 Contact & Portfolio
👤 **Jiaxin(Berry) Tian**  
📧 berrytian15@gmail.com  
🔗 https://github.com/berrrrry-hub?tab=repositories
