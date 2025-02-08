## Project Title: Employee Attrition Prediction Using Supervised Machine Learning
## Team Members:
1.	Anthony Lopez
2.	Emilia Elangwe
3.	Thomas Sullivan
4.	Iram Anwar

   
The objective of this project is to build a supervised machine learning model to predict employee attrition using the IBM HR Analytics dataset. The project will also focus on utilizing advanced data visualization techniques to interpret model results and present actionable insights that can help organizations reduce turnover rates which can have significant financial and operational impacts on organizations. Understanding the factors that contribute to attrition and predicting which employees are at risk of leaving can help HR departments implement targeted retention strategies. By applying supervised machine learning models and data visualization, this project aims to provide a data-driven approach to managing employee retention.
## Dataset Description:
### Source: IBM HR Analytics Employee Attrition & Performance Dataset
[Data Link](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)


In this project, the goal was to build a machine learning model to predict employee attrition using the IBM HR dataset. The objective was to identify patterns that lead to employees leaving the organization and improve employee retention strategies. Three classification models were considered: Logistic Regression, Decision Tree, and Random Forest. The performance of these models was evaluated using accuracy, cross-validation scores, and hyperparameter optimization techniques.
## Methodology:
#### 1. Data Preprocessing: 
Categorical variables were encoded, and continuous variables were scaled using StandardScaler to prepare the data for model training.
#### 2. Model Selection:
Logistic Regression, Decision Tree, and Random Forest were chosen due to their applicability to classification tasks and their ability to handle different data structures.
#### 3. Model Training and Cross-Validation: 
Models were trained using the training data and evaluated using 5-fold cross-validation. Initial evaluation showed that Random Forest had the highest accuracy on the training data, but it suffered from overfitting.

## Results:
The Random Forest model initially showed the highest training accuracy but was found to be overfitting, with a significant drop in test accuracy. Logistic Regression, despite its lower performance on training data, had a more balanced performance on test data. After applying hyperparameter tuning and scaling, the Random Forest model achieved the best cross-validation accuracy of 86%.
## Conclusion:
The Random Forest Model, after scaling and optimization, performed the best across all evaluation metrics. It is the recommended model for predicting employee attrition in this context. Future work could focus on additional feature engineering or exploring advanced ensemble models.



