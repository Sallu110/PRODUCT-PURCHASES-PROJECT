# PRODUCT-PURCHASES-PROJECT


# Predict Product Purchases for Bank Telemarketing Campaign
This project aims to predict product purchases based on a bank telemarketing dataset using machine learning techniques. Specifically, the Random Forest Classifier algorithm is employed to handle the categorical dataset. Additionally, Recursive Feature Elimination (RFE) is implemented to enhance the model by selecting the most important features.

# "steps of project" 
Introduction
data preprocessing 
Model Training
Feature Selection
Evaluation
Results
Conclusion


# Introduction
In this project, i predict whether a client will purchase a product based on various features provided in a bank telemarketing dataset. The Random Forest Classifier is chosen due to its effectiveness in handling categorical data. i also apply Recursive Feature Elimination (RFE) to improve the model by selecting the most significant features.

# data preprocessing
# Import Libraries
First, we import the necessary libraries:

# import pandas as pd
# Read the Dataset
Load the dataset from a CSV file:
f = pd.read_csv('bank.csv')

# Drop the 'duration' Feature
The 'duration' feature is dropped as it is not used for predictive modeling. It can be misleading because it is related to the outcome of the call:
f = f.drop('duration', axis=1)

# Create Features (X) and Target Variable (Y)
Split the data into features (X) and the target variable (Y):
X = f.iloc[:, :-1]
Y = f.iloc[:, -1]

# Convert Categorical Variables
Convert categorical variables into dummy/indicator variables. This step is crucial as machine learning models require numerical input:
X = pd.get_dummies(X, drop_first=True)
Y = pd.get_dummies(Y, drop_first=True)

# Train-Test Split
Split the dataset into training and testing sets. We use stratified sampling to ensure that the target variable's distribution is similar in both training and testing sets:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, str

# Feature Selection
-Recursive Feature Elimination (RFE): 
Apply RFE to select the top 30 features.
-Transform Data: 
Transform the training and testing sets using the selected features.

# Evaluation
-Confusion Matrix: 
Evaluate the model using a confusion matrix.
- Model Score: 
Calculate the accuracy score of the model.

# Results
After training and evaluating the model, the results include:

- Confusion Matrix: 
Displays the performance of the model.
- Accuracy Score: 
Indicates the model's accuracy on the test set.
- Feature Ranking:
Lists the importance and ranking of each feature.

# Conclusion
This project demonstrates the use of Random Forest Classifier and Recursive Feature Elimination for predicting product purchases in a bank telemarketing campaign. The model is evaluated and significant features are identified, enhancing the overall performance.




