#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:09:27 2018

@author: GR5293 Group4 Project Code
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


# Import training set and test set
X_train = pd.read_csv('X_train_dummy.csv').iloc[:,2:]
X_test = pd.read_csv('X_test_dummy.csv').iloc[:,2:]
y_train = pd.read_csv('y_train_dummy.csv', header=None).iloc[:,1:]
y_test = pd.read_csv('y_test_dummy.csv', header=None).iloc[:,1:]


# LASSO

## scaled the data
X_scaled_train = preprocessing.scale(X_train)
X_scaled_test = preprocessing.scale(X_test)

## tuning alphas
alphas=(0.001,0.002,0.003,0.005,0.01,0.015,0.02)
scores=[]
rss=[]
for alpha in alphas:
    regr = Lasso(alpha=alpha)
    # Train the model using the training sets
    regr.fit(X_scaled_train, y_train)
    score = np.mean(cross_val_score(regr, X_scaled_train, y_train, cv=5))
    scores.append(score)

## plot cross validation results    
plt.plot(alphas,scores)
plt.xlabel('lambda')
plt.ylabel('R^2')
plt.title('Performance for different lamda with 5 folds cross validation' )
plt.show()

## predict by best model
best_alpha=alphas[np.argmax(scores)]
best_alpha
regr = Lasso(alpha=best_alpha)
regr.fit(X_scaled_train, y_train)
pred_train= regr.predict(X_scaled_train)
pred_test= regr.predict(X_scaled_test)

## r2 & mse
mse_train= np.mean((pred_train-np.array(y_train))**2)
mse_test= np.mean((pred_test-np.array(y_test))**2)
score_train= regr.score(X_scaled_train, y_train)
score_test= regr.score(X_scaled_test, y_test)

## feature extraction
[round(x,2) for x in range(len(regr.coef_))]
df_coeffs = pd.DataFrame({'coeffs':abs(regr.coef_), 'name':X_train.columns.values})
df_coeffs=df_coeffs.sort_values(['coeffs'])
df_coeffs=df_coeffs[::-1]
df_coeffs[0:10].plot(x='name',y='coeffs',kind='bar')
plt.show()
df_coeffs[0:10]


# Decision Tree

## Use library algorithms DecisionTreeRegressor + Cross Validation
## Run algorithm on trainning set to generate and store prediction algorithm.
## tuning max_depth
depth = range(1,31)
scores=[]
for i in depth:
    regr = DecisionTreeRegressor(max_depth=i)
    # Train the model using the training sets
    regr.fit(X_train, y_train)
    score = np.mean(cross_val_score(regr, X_train, y_train, cv=5))
    scores.append(score)

## plot cross validation results   
plt.plot(depth,scores)
plt.xlabel('max_depth')
plt.ylabel('R^2')
plt.title('Performance for different max_depth with 5 folds cross validation' )
plt.show()

## predict by best model
regr = DecisionTreeRegressor(max_depth=7)
regr.fit(X_train, y_train)
y_predict_tree = regr.predict(X_test)
## r2 & mse
r2 = r2_score(y_test, y_predict_tree)
mse = mean_squared_error(y_test, y_predict_tree)

## feature extraction
columns_name = list(X_train)
importances = regr.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

importance = []
name = []
for f in range(10):
    print(columns_name[indices[f]], importances[indices[f]])
    importance.append(importances[indices[f]])
    name.append(columns_name[indices[f]])

df_coeffs = pd.DataFrame({'importance':importance, 'name':name})
df_coeffs.plot(x='name',y='importance',kind='bar')
plt.show()


# Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
depth = range(1,10)
num = range(50,100)
scores=[]
depthnum=[]
treenum=[]
for i in depth:
    for j in num:
        regr = RandomForestRegressor(max_depth=i,n_estimators = j)
    # Train the model using the training sets
        regr.fit(X_train, y_train)
        score = np.mean(cross_val_score(regr, X_train, y_train, cv=5))
        scores.append(score)
        depthnum.append(i)
        treenum.append(j)
        
# Choose the best parameters        
forest_ma = np.matrix([scores, depthnum,treenum])
forest = pd.DataFrame(forest_ma)
for i in range(forest.shape[1]):
    if forest.loc[0,i] == max(forest.loc[0]):
        print(forest.loc[1,i],forest.loc[2,i])


regressor = RandomForestRegressor(n_estimators = 92, random_state = 0,max_depth=9)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_randomF = regressor.predict(X_test)
# MSE&R2
mean_squared_error(y_test, y_pred_randomF)
r2_score(y_test, y_pred_randomF) 

## feature extraction
from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(regressor, prefit=True)
X_new = model.transform(X_train)

columns_name = list(X_train)
importances = regressor.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

importance = []
name = []
for f in range(10):
    print(columns_name[indices[f]], importances[indices[f]])
    importance.append(importances[indices[f]])
    name.append(columns_name[indices[f]])
    
plt.xticks(rotation='90')
sns.barplot(x=name, y=importance)
plt.xlabel('name of important features', fontsize=15)
plt.ylabel('Percentage of important features', fontsize=15)
plt.title('Important Features', fontsize=15)
