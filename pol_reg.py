#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 09:21:47 2023

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
#polynomial regression 
pol_points = pd.read_csv("polynomial.csv")
plt.scatter(pol_points.x, pol_points.y)
degree = 2 
polynomial_features = PolynomialFeatures(degree = degree)
x = pol_points.x[0:6, np.newaxis]
y = pol_points.y[0:6, np.newaxis]

print(x)
x_poly = polynomial_features.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)
plt.scatter(x, y)
plt.plot(x, y_poly_pred)
print(x_poly)
#calculating r_squared 
print(model.score(x_poly, y))
