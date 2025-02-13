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



## **Project Overview**  
This project aims to predict employee attrition using machine learning techniques on the IBM HR dataset. By analyzing various employee-related factors, we compare different supervised learning models to determine the most effective approach for attrition prediction.  

## **Dataset**  
The dataset contains various employee attributes, such as:  
- **Demographics** (Age, Gender, Marital Status)  
- **Job Information** (Department, Job Role, Job Satisfaction)  
- **Compensation & Benefits** (Salary, Stock Options, Work-Life Balance)  
- **Performance Metrics** (Overtime, Training Hours, Years at Company)  
- **Attrition Label** (Yes/No - Target Variable)  

## **Machine Learning Models Used**  
We applied and evaluated the following models:  
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  

## **Methodology**  
The project follows a structured pipeline:  
1. **Data Preprocessing**  
   - Handling missing values  
   - Encoding categorical variables  
   - Standardizing numerical features  
2. **Model Training & Evaluation**  
   - Training models on unscaled data  
   - Implementing feature selection and standardization  
   - Hyperparameter tuning using **GridSearchCV**  
3. **Performance Metrics**  
   - **Accuracy Score**  
   - **Confusion Matrix**  
   - **Classification Report (Precision, Recall, F1-score)**  

## **Results & Findings**  
### **Model Performance Summary**  
| Model                             | Mean Accuracy | Training Accuracy | Test Accuracy |
|-----------------------------------|---------------|-------------------|---------------|
| **Logistic Regression (Unscaled)**    | 0.8408        | 0.8403            | 0.8424        |
| **Decision Tree (Unscaled)**          | 0.7762        | 1.0000            | 0.7554        |
| **Random Forest (Unscaled)**          | 0.8585        | 1.0000            | 0.8478        |
| **Logistic Regression (Scaled & Selected)** | 0.8408        | 0.8838            | 0.3587        |
| **Decision Tree (Scaled & Selected)** | 0.7776        | 1.0000            | 0.7609        |
| **Random Forest (Scaled & Selected)** | 0.8612        | 1.0000            | 0.8397        |
| **Logistic Regression (Grid Search)** | 0.8739        | 0.8838            | 0.8668        |
| **Decision Tree (Grid Search)**       | 0.8394        | 0.8884            | 0.8668        |
| **Random Forest (Grid Search)**       | 0.8602        | 0.9501            | 0.8668        |

### **Key Insights**
- **Unscaled models suffered from overfitting**, especially Decision Tree and Random Forest (Training Accuracy = 1.0000).  
- **Feature scaling impacted Logistic Regression negatively**, leading to a **significant drop in test accuracy**.  
- **Hyperparameter tuning (Grid Search) improved model generalization**, achieving a stable **test accuracy of 86.68% across all models**.  
- **Random Forest had the highest training accuracy after tuning (0.9501), but Logistic Regression remained competitive with better interpretability**.  

## **Technologies & Tools Used**
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn  
- **Machine Learning Techniques**: Feature Scaling, One-Hot Encoding, Hyperparameter Tuning (GridSearchCV)  
- **Visualization**: Matplotlib, Seaborn  
- **Evaluation Metrics**: Accuracy Score, Confusion Matrix, Classification Report  

## **Conclusion**  
This project demonstrates the importance of data preprocessing, feature engineering, and hyperparameter tuning in **building effective machine learning models for employee attrition prediction**. The **Random Forest and Logistic Regression models performed best** after tuning, offering a balance between accuracy and interpretability.  This project offers quality insight on the analytics behind business culture and the major factors that drive employees to stay with or leave a company.  Employee attrition and retention will always play an important role in the success of businesses across the globe and being able to analyze and potentially predict its trends could help these businesses adapt and thrive.





