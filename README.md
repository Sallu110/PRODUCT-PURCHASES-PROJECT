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

# Random forest classifier 

from sklearn.ensemble import RandomForestClassifier

# TRAIN THE MODEL 

rfc = RandomForestClassifier(random_state = 1234)
rfc.fit(X_train,Y_train)
R_predict = rfc.predict(X_test)

# Evaluate the model 
from sklearn.metrics import confusion_matrix 
cm    = confusion_matrix(Y_test,R_predict)
score = rfc.score(X_test,Y_test) 

cm stands for confusion matrix

![Screenshot 2024-07-18 180911](https://github.com/user-attachments/assets/d0defb6c-c326-441e-9937-18fd0cd6afef)


# Feature Selection
-Recursive Feature Elimination (RFE): 
Apply RFE to select the top 30 features.

# from sklearn.feature_selection import RFE

rfc2 = RandomForestClassifier()
rfe2 = RFE(estimator = rfc2,n_features_to_select=30,step = 1)
rfe2.fit(X,Y)
X_train_rfe = rfe2.transform(X_train)
X_test_rfe = rfe2.transform(X_test)

# fit the random forest classifer to the new train and test reduced feature data 

rfc2.fit(X_train_rfe,Y_train)

# test the model with new data 

R_predict2 = rfc2.predict(X_test_rfe)

# score and evaluate the model  

cm2   = confusion_matrix(Y_test,R_predict2)
score2 = rfc2.score(X_test_rfe,Y_test)   

cm2 stands for confusion matrix2

![Screenshot 2024-07-18 180930](https://github.com/user-attachments/assets/1d4be301-9282-47f4-a2dd-d31caaecf6fb)


# get the columns names 

columns = list(X.columns)

# get the ranking of the features
# ranked 1 for selected feature 

ranking = rfe2.ranking_

# get the feature importance 

feature_importance = rfc.feature_importances_

# creat the data fram of features selected, features ranking and features importance


rfe_selected = pd.DataFrame()

dataframe1 = pd.DataFrame(columns)
dataframe2 = pd.DataFrame(ranking)
dataframe3 = pd.DataFrame(feature_importance)
rfe_selected = pd.concat([dataframe1,dataframe2,dataframe3],axis =1)
rfe_selected.columns = ['columns','Ranking','feature_importance']



# Results
After training and evaluating the model, the results include:

- Confusion Matrix: 
Displays the performance of the model.
- Accuracy Score: 
Indicates the model's accuracy on the test set.
- Feature Ranking:
Lists the importance and ranking of each feature.

![Screenshot 2024-07-18 181213](https://github.com/user-attachments/assets/dd0a6d9a-fa94-4796-a225-6d5d68a03a74)

![Screenshot 2024-07-18 181237](https://github.com/user-attachments/assets/ab7d55b5-fcf8-4a31-a1b1-098f8374dd0a)

![Screenshot 2024-07-18 181257](https://github.com/user-attachments/assets/dfe126fc-b7b1-4ab7-9355-8498622e7478)




# Conclusion
This project demonstrates the use of Random Forest Classifier and Recursive Feature Elimination for predicting product purchases in a bank telemarketing campaign. The model is evaluated and significant features are identified, enhancing the overall performance.




