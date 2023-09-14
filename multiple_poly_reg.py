#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 12:09:59 2023

@author: anoushkajawale
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

housing = fetch_california_housing()
print(housing.feature_names)
print(housing.DESCR)

#the festure names don't include the target 
data = pd.DataFrame(housing.data, columns=housing.feature_names)
print(data)

#so we have to add a column for the target 
data['MedHouseVal'] = housing.target
print(data)

#to get datatypes of data columns
print(data.info())
print(data.isnull().sum())

#choosing features through correlation 
corr = data.corr()
print(corr)

#getting two largest correlations positive
print(data.corr().abs().nlargest(3, "MedHouseVal").index)
#above gave Index(['MedHouseVal', 'MedInc', 'AveRooms'], dtype='object')
print(data.corr()[data.corr().abs().nlargest(3, "MedHouseVal").index].nlargest(3, "MedHouseVal"))

#plot showingm relationship between MedInc and MedHouseValue
#plt.scatter(data['MedInc'], data['MedHouseVal'])

"""plt.scatter(data['AveRooms'], data['MedHouseVal'], marker='x')
plt.xlabel("AveRooms")
plt.ylabel("MedHouseVal")"""


print(data['AveRooms'].max())

#splitting data 70/30 for test and training ata 
# 70 for training 
x = np.c_[data['MedInc'], data['AveRooms']]
y = data['MedHouseVal']
print("This is y")
print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

degree = 5
polynomial_features = PolynomialFeatures(degree=degree)
x_train_poly = polynomial_features.fit_transform(x_train)
print(x_train_poly)











