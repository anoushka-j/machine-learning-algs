#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:58:28 2023

@author: anoushkajawale
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

cancer = load_breast_cancer()

#---copy from dataset into a 2-d list---
X = []

#nested loop
for target in range(2):
    X.append([[], []])
    for i in range(len(cancer.data)):              # target is 0 or 1
        if cancer.target[i] == target:
            X[target][0].append(cancer.data[i][0]) # first feature - mean radius
            X[target][1].append(cancer.data[i][1]) # second feature — mean texture
 
colours = ("r", "b")   # r: malignant, b: benign
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
for target in range(2):
    ax.scatter(X[target][0],
               X[target][1],
               c=colours[target])
 
ax.set_xlabel("mean radius")
ax.set_ylabel("mean texture")
plt.show() 


log_regression = LogisticRegression()
#getting mean radius 
x = cancer.data[:, 0]
#print(x)
#getting “0: malignant, 1: benign”
y = cancer.target

log_regression.fit(X = np.array(x).reshape(len(x), 1), y = y)
#print(X)


#trained model intercept 
print(log_regression.intercept_[0])

#trained model coefficient
print(log_regression.coef_)

#knowing these two values allows plotting of sigmoid curve that fits the points 



#MAKING PREDICTIONS
print(log_regression.predict_proba([[30]]))
print(load_breast_cancer().DESCR)











