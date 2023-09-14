#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 09:50:23 2023

@author: anoushkajawale
"""

from sklearn import datasets, svm 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
iris = datasets.load_iris()
print(iris.feature_names)
target = iris.target
print(iris.target_names)

values, counts = np.unique(target, return_counts=True)
for value, count in zip(values, counts) : 
    print(f"{value} : {count}")

#only going to use first two features of dataset 

x = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)


#---plot the points---

colors = ['red', 'green', 'blue']
for color, i, target in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(x[y==i, 0], x[y==i, 1], color=color, label=target)
 
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend(loc='best', shadow=False, scatterpoints=1)
 
plt.title('Scatter plot of Sepal width against Sepal length')
plt.show()

#defining param grid to find best value of C
param_grid= {'C': [1, 3, 0.9, 0.8, 0.95]}
param_gamma = {'gamma': [i for i in range(0, 10)]}
param_gamma_and_deg = {'gamma': [1], 
                       'degree': [1, 2, 3, 4, 5]}

#perform grid search with cross validation to find best value of C
#process automated with GridSearchCV
"""
grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(best_params)
print(best_model)
accuracy = best_model.score(X_test, y_test)
print(accuracy)
# use the SVC (Support Vector Classification) 
#linear kernel 
C = 1
clf = svm.SVC(kernel='linear', C=C).fit(x, y)
preds = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(accuracy)

#print(np.unique(preds, return_counts=True))
"""
#Radial Basis Function Kernel
C = 1
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_gamma, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(best_params)
print(best_model)
clf = svm.SVC(kernel='rbf', gamma=1, C=C).fit(x, y)
preds = clf.predict(x)
accuracy = clf.score(X_test, y_test)
print(accuracy)
print(np.unique(preds, return_counts=True))


#Polnomial Kernel 
C = 1 
"""grid_search = GridSearchCV(svm.SVC(kernel='poly', degree=2), param_gamma_and_deg, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(best_params)
print(best_model)
clf = svm.SVC(kernel='poly', degree=1, C=C, gamma=1).fit(x, y)
preds = clf.predict(x)
accuracy = clf.score(X_test, y_test)
print(accuracy)
#print(np.unique(preds, return_counts=True))"""



