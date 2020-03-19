# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:51:58 2019

@author: Ashir
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#cross validation doesnt works so i have to use sklearn.model_selection
from sklearn.model_selection import train_test_split
import pandas as pd

#dataset
mydataset = pd.read_csv('data1.csv')
X = mydataset.iloc[:,:-1].values
y = mydataset.iloc[:,1].values

#Spliting the dataset into Training set and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

#X_train= X_train.reshape(-1, 1)
#y_train= y_train.reshape(-1, 1)
#X_test = X_test.reshape(-1, 1)
#y_test = y_test.reshape(-1,1)


#Model Initialization
regression_model = LinearRegression()

#fit the data
regression_model.fit(X_train,y_train)

#prediction
y_predict = regression_model.predict(X_test)
X_predict = regression_model.predict(X_train)

#model evaluation
rmse = mean_squared_error(y_test,y_predict)
r2 = r2_score(y_test,y_predict)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

#plot model on training data
plt.scatter(X_train,y_train,color = 'red')
plt.ylabel('Sales of potato')
plt.xlabel('Months')
plt.plot(X_train, X_predict)
plt.show()

#test data comparison
df_test = pd.DataFrame({'Month': X_test.flatten(),'Actual': y_test.flatten(), 'Predicted': y_predict.flatten()})
print(df_test)
#plot on test data
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_predict, color='red', linewidth=2)
plt.show()

#train data comparison
df_train = pd.DataFrame({'Month': X_train.flatten(),'Actual': y_train.flatten(), 'Predicted': X_predict.flatten()})
print(df_train)
