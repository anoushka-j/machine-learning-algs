#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 09:37:08 2023

@author: anoushkajawale
"""

import numpy as np 
import pandas as pd 
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
data = iris.data
#take the first two features, all rows and first 2 columns
X = data[:, :4]
y = iris.target

k = 1 
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)
cv_scores = []
folds = 10 
a = pd.Series([1, 2, 3, 4, 5])
print(a)





