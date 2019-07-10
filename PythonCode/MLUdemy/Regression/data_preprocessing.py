# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#import library

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


#import data set 
dataset= pd.read_csv('Data.csv')
##create matrix of features
X=dataset.iloc[:,:-1].values 
##create the dependent variable vector (output)
Y=dataset.iloc[:,-1].values 

#missing values
NaN_column=dataset.isnull().any()[:-1]
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,NaN_column])
X[:,NaN_column] = imputer.transform(X[:,NaN_column])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
#? should we scale the dummies variables ?
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(y_train.reshape(-1,1))
