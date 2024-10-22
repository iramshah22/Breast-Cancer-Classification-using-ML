# **Breast Cancer Classification using Machine Learning**

# Overview
This project focuses on building a machine learning model to classify whether a tumor is malignant or benign based on the characteristics of cell nuclei present in a breast cancer dataset. Using a logistic regression model, we aim to predict the type of cancer from the provided features and assess the accuracy of our predictions.

# Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, available in the sklearn library. It contains 569 instances and 30 features, which describe the characteristics of the cell nuclei computed from digitized images of a fine needle aspirate (FNA) of a breast mass. The target variable is binary, indicating whether the tumor is malignant (1) or benign (0).

# Features
The dataset includes the following features:
Mean radius
Mean texture
Mean perimeter
Mean area
Mean smoothness
And 25 additional attributes related to the cell nuclei.

# Process
Data Collection: The dataset is loaded using the sklearn.datasets module.
Data Preprocessing: The dataset is split into training and testing sets to ensure unbiased evaluation of the model’s performance.
Modeling: Logistic Regression is applied to the training data to build the classification model.
Evaluation: The model’s accuracy is assessed using the test data, and metrics such as accuracy score are calculated to measure its performance.

# Results
The accuracy of the logistic regression model on this dataset is evaluated to ensure its reliability in distinguishing between malignant and benign tumors.

# Technologies Used
Python
NumPy
Pandas
Scikit-learn

# How to Run the Code
Clone the repository.
Install the necessary libraries using pip install -r requirements.txt.
Run the Jupyter Notebook or Python script to execute the code.

# Conclusion
This project demonstrates the use of logistic regression for binary classification, specifically in identifying breast cancer malignancy. Further enhancements can include testing different models or tuning hyperparameters to improve the performance.
