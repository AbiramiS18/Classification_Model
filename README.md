# Telecom Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn for a telecom company. "Churn" refers to the phenomenon where customers stop doing business with a company. By analyzing customer demographics, services, and billing information, this machine learning model predicts whether a customer is likely to leave (Churn: Yes/No).

This solution is implemented using **Python** in a Jupyter Notebook, utilizing pipelines for preprocessing and model training.

## Dataset
The project uses the **Telco Customer Churn** dataset.
* **Target Variable:** `Churn` (Binary: Yes/No)
* **Features:** Customer demographics (Gender, Senior Citizen), Services (Phone, Internet, Tech Support), and Account info (Contract type, Payment method, Monthly Charges).

## Tech Stack & Libraries
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Model Persistance:** Joblib

## Key Steps in the Notebook
1.  **Data Cleaning:**
    * Handled missing values in `TotalCharges`.
    * Converted categorical variables for analysis.
2.  **Exploratory Data Analysis (EDA):**
    * Visualized churn rates by Contract Type and Gender.
    * Correlation heatmaps to identify key drivers of churn.
3.  **Data Preprocessing (Pipeline):**
    * **Numerical:** Standard Scaling.
    * **Categorical:** One-Hot Encoding.
4.  **Model Training:**
    * Tested three algorithms:
        * Logistic Regression
        * Random Forest Classifier
        * XGBoost Classifier
5.  **Hyperparameter Tuning:**
    * Used `GridSearchCV` to optimize model parameters for accuracy and F1-score.
6.  **Evaluation:**
    * Evaluated models using Confusion Matrix, Precision, Recall, and ROC-AUC scores.

## How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
    ```
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Churn_Prediction.ipynb
    ```
