# Breast Cancer Classification using Logistic Regression

## Objective
This project demonstrates the implementation of a machine learning model to predict whether a breast tumor is malignant or benign based on diagnostic measurements. The model is built using Logistic Regression, a powerful and interpretable classification algorithm.

## Dataset
The model is trained on the Wisconsin Breast Cancer dataset. This dataset contains 30 numerical features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The goal is to classify tumors into two categories: Malignant (M) or Benign (B).

## Project Workflow
1.  **Data Preprocessing:**
    * Loaded the dataset and dropped irrelevant columns like 'id'.
    * The categorical target variable, 'diagnosis', was encoded into a binary numerical format (M=1, B=0).

2.  **Feature Scaling:**
    * Applied `StandardScaler` to all features to normalize their scales. This is a crucial step to ensure that features with larger ranges do not disproportionately influence the model's predictions.

3.  **Model Training:**
    * The dataset was split into an 80% training set and a 20% testing set.
    * A Logistic Regression model was trained on the scaled training data.

4.  **Model Evaluation:**
    * The model's performance was rigorously evaluated on the unseen test data using:
        * **Accuracy Score:** To measure the overall correctness of predictions.
        * **Classification Report:** To analyze precision, recall, and F1-score for each class.
        * **Confusion Matrix:** To visualize the model's performance in distinguishing between malignant and benign cases.

## Tools and Libraries
* Python
* Pandas
* Scikit-learn
* Matplotlib & Seaborn
