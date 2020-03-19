# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 22:54:34 2018

@author: Ashir
"""

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn import datasets, linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_val_score


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print(train.shape,test.shape)
#print(train.ndim,test.ndim)

#print(train.head(2))
#print(train.dtypes)

#print(train.describe())
#print(train.columns.values)

#print(train.info())

X = np.array(train.drop(['label'],1).astype(int))
y = np.array(train['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#p : integer, optional (default = 2)
#metric : string or callable, default ‘minkowski’
# The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric
knne = neighbors.KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knne.fit(X_train,y_train)
accuracy = knne.score(X_test, y_test)  
print("Accuracy using eucledean  :: "+str(accuracy))


#for cosine
knnc = neighbors.KNeighborsClassifier(n_neighbors=5, metric='cosine')
knnc.fit(X_train,y_train)
accuracy = knne.score(X_test, y_test)  
print("\n Accuracy using cosine  :: "+str(accuracy))


#predictions using knn euclidean
y_prede = knne.predict(X_test)

#predictions using knn cosine
y_predc = knnc.predict(X_test)

#report for eucledean
print("\n Confusion Matrix: \n"+str(confusion_matrix(y_test, y_prede)))  
print("\n Classification Report: \n"+str(classification_report(y_test, y_prede)))

#report for cosine
print("\n Confusion Matrix: \n"+str(confusion_matrix(y_test, y_predc)))  
print("\n Classification Report: \n"+str(classification_report(y_test, y_predc)))

#error calculation for euclidean
error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    print("\n completed "+str(i))
#figure    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')      

