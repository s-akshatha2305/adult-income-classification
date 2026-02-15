# Adult Income Classification using Machine Learning
---

## 1. Problem Statement

The goal of this project is to develop ML models to predict whether an individual's income exceeds $50K per year based on demographic and features such as age, education, occupation, and working hours.

This is a binary classification problem where the target variable is:
<=50K or >50K

---

## 2. Dataset Description

The dataset is Adult Income Dataset from Kaggle.

- Number of observations: 48,842
- Number of features: 14 input features + 1 target variable
- Target variable: Income (<=50k or >50k)
- Different features are age, workclass, education, marital status, occupation, relationship etc.

### Data Preprocessing:
- Replaced '?' with null 
- Label encoding applied for categorical columns
- Standard scaling performed for numerical columns

---

## 3. Models Used

The following machine learning models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

## 4. Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Results:

| Model          | Accuracy | Precision | Recall   | F1       | MCC      | AUC      |
|----------------|----------|-----------|----------|----------|----------|----------|
| Logistic       | 0.821117 | 0.711497  | 0.446664 | 0.548801 | 0.463166 | 0.848527 |
| KNN            | 0.827640 | 0.665468  | 0.587835 | 0.624247 | 0.514636 | 0.854533 |
| Random Forest  | 0.862355 | 0.814717  | 0.562869 | 0.665772 | 0.598518 | 0.914548 |
| Decision Tree  | 0.805196 | 0.597008  | 0.615978 | 0.606345 | 0.477064 | 0.741022 |
| XGBoost        | 0.871531 | 0.777009  | 0.662733 | 0.715336 | 0.636396 | 0.926523 |
| Naive Bayes    | 0.798894 | 0.678439  | 0.331366 | 0.445258 | 0.372291 | 0.852216 |

---

## 5. Model Comparison Table

| Model               | Performance |
|---------------------|-------------|
| Logistic Regression | Model has good accuracy and AUC but moderate recall (misses high-income individuals) |
| KNN                 | Model provides better recall and balanced performance, but lower precision |
| Random Forest       | Strong model with high accuracy and good precision-recall balance, makes it more reliable |
| Decision Tree       | Balanced recall, moderate performance but less precision and AUC than other models |
| Naive Bayes         | Weak performer of all models with low recall and F1 score, poorly detects positive cases but fast |
| XGBoost             | Best model with high accuracy, F1 score, AUC. Provides most balanced predictions |

---

## 6. Overall Observations

- XGBoost demonstrated the best overall performance, achieving the highest accuracy, F1 score, MCC, and AUC.
- Random Forest also performed strongly, showing good generalization and consistent results.
- Logistic Regression delivered moderate performance, suggesting the dataset is only partially linearly separable.
- Decision Tree exhibited slightly lower performance compared to ensemble methods, likely due to overfitting.
- Naive Bayes showed the weakest performance, mainly because of its assumption of feature independence.
- Overall, ensemble models such as Random Forest and XGBoost outperformed individual models.

## 7. Capabilities of the app

- Download test file and Upload it
- Cached data loading for performance
- Preview dataset with expandable stats
- Handles missing values and encoding automatically
- Model Evaluation (6 models): Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
- Comprehensive metrics: Accuracy, Precision, Recall, F1, MCC, AUC

- Visualizations:
1. Performance bar charts for quick metric overview
2. Confusion matrix (values + heatmap)
3. Error analysis with pie charts (correct/incorrect, TP/TN/FP/FN)
4. Top 5 feature importance rankings

- Model Comparison:
1. One-click evaluation of all 6 models
2. Progress tracking with status updates
3. individual metric bar charts
4. Grouped bar chart for all metrics
5. Interactive heatmap visualization
6. Automatic best model identification