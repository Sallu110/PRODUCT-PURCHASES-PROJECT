

# predict product purchase for the banktelemarketing campaign 
# implement recursive feature elemination 

# import libraries 

import pandas as pd 

# read file 

f = pd.read_csv('bank.csv')

# drop the duration deature 

f = f.drop('duration', axis  =1)

X = f.iloc[:,:-1]
Y = f.iloc[:,-1]

X = pd.get_dummies(X, drop_first = True)
Y = pd.get_dummies(Y, drop_first = True)

# create the X and Y variables 


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1234, stratify=Y)

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


# APPLY  RECURSIVE FEATURE ELEMINATION 

from sklearn.feature_selection import RFE

rfc2 = RandomForestClassifier()
rfe2 = RFE(estimator = rfc2,n_features_to_select=30,step = 1)

rfe2.fit(X,Y)



X_train_rfe = rfe2.transform(X_train)
X_test_rfe = rfe2.transform(X_test)

#fit the random forest classifer to the new train and test reduced feature data 

rfc2.fit(X_train_rfe,Y_train)

# test the model with new data 

R_predict2 = rfc2.predict(X_test_rfe)

# score and evaluate the model  

cm2   = confusion_matrix(Y_test,R_predict2)
score2 = rfc2.score(X_test_rfe,Y_test)        


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

# PROJECT END 