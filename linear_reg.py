#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 09:57:27 2023

@author: anoushkajawale
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import pickle

iris_data = datasets.load_iris()
irises = pd.DataFrame(iris_data['data'])
print(irises)

# represents the heights of a group of people in meters
heights = [[1.6], [1.65], [1.7], [1.73], [1.8]]
 
# represents the weights of a group of people in kgs
weights = [[60], [65], [72.3], [75], [80]]
 
plt.title('Weights plotted against heights')
plt.xlabel('Heights in meters')
plt.ylabel('Weights in kilograms')
 
plt.plot(heights, weights, 'k.')
 
# axis range for x and y
plt.axis([0, 1.85, -200, 200])
plt.grid(True)


#create and fit the model 
model = LinearRegression()
model.fit(X=heights, y=weights)
extreme_heights = [[0], [1.8]]
plt.plot(extreme_heights, model.predict(extreme_heights), color='pink')
weight_pred = model.predict([[1.75]])[0][0]
print(weight_pred)

#another way to get the y-int
print(model.predict([[0]]))

#getting the gradient/slope
print(model.coef_) #gives [[103.31046312]]
print(model.coef_[0][0]) #gives 103.31046312178385 (basically indexed for a cleaner version)

#what if we want to round?
print(round(model.coef_[0][0], 2)) #gives 103.31

#calculating residual sum of squares 
print(np.sum(weights - model.predict(heights)) **2 )
print('Residual sum of squares: %.2f' %
       np.sum((weights - model.predict(heights)) ** 2))


#calculating TSS on the following test data 
heights_test = [[1.58], [1.62], [1.69], [1.76], [1.82]]
weights_test = [[58], [63], [72], [73], [85]]

test_weights_mean = np.mean(weights_test)
TSS = np.sum(np.ravel(weights_test - test_weights_mean)**2)


print(TSS)

#RSS 

RSS = np.sum(
    (
     np.ravel(weights_test) - 
     np.ravel(model.predict(heights_test))
     
     ) ** 2)

print(RSS)

#R squares method 
R_squared = 1 - (RSS/TSS)
print(R_squared)

#BUT the R_squared doesn't have to be calculated manually 
print(model.score(heights_test, weights_test))


#PERSISTING THE MODEL 

pickle.dump(model, open("height-weight-model.sav", 'wb'))

#how to load the saved model?
loaded_model = pickle.load(open("height-weight-model.sav", 'rb')) 
result = loaded_model.score(heights_test, weights_test)
print(result)


print(irises.isnull().sum())

