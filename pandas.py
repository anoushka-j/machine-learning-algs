#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:30:17 2023

@author: anoushkajawale
"""

import pandas as pd
import numpy as np
import math


series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'c'])
print(series)
print(series[2])
print(series.iloc[2])
print(series['c'])
print(series[1:])

#KeyError: "Cannot get left slice bound for non-unique label: 'c'"
#print(series.loc['c': ])

#datetime range as the index of a series 
dates1 = pd.date_range('2003-04-03', periods=5, freq='M')
print(dates1)

#assigning datetime range as an index to a series
series.index = dates1
print(series)


#DATAFRAMES 
df = pd.DataFrame(np.random.randn(10,4), columns=['b', 'a', 'C', 'D'])


#reading from csv
data = pd.read_csv("data.csv")
#gets first 5 rows 
print(data.head())
print(data.describe())

#common prpblem is to check if a DF or S
def checkType(data) : 
    if isinstance(data, pd.DataFrame) : 
        return 'Dataframe'
    if isinstance(data, pd.Series) : 
        return 'Series'


print(checkType(data))

#sorting the df columns 
print(df.sort_index(axis=0, ascending=False))
print(df.sort_values('b', axis=0, ascending=False))

sq_root = lambda x : x*0
add_two = lambda x : x + 100
positive_root = lambda x : math.sqrt(x) if x > 0 else 0

#to directly apply to the df
df = df.apply(add_two)
#to apply positive_root lambda function,have to loop 
#through the columns


for column in df : 
    df[column] = df[column].apply(positive_root)
    


#removing columns and rows 
students = {'name': ['Janet', 'Nad', 'Timothy', 'June', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [6, 13, 14, 1, 7]}


students = pd.DataFrame(students, index =
       ['Singapore', 'China', 'Japan', 'Sweden', 
        'Norway'])

#can also remov based on a condition
print(students[students['name'] != "Amy"])

#removes second row 
print(students.drop(students.index[1]))

#removing columns, axis=1 for columns
print(students.drop('name', axis=1))
print(students.drop(students.columns[2], axis=1))

#generating a corsstab
df = pd.DataFrame(
     {
        "Gender": ['Male','Male','Female','Female','Female'],
        "Team"  : [1,2,3,3,1]
     })
print(df)
print("Displaying the distribution of genders on each team.")

print(pd.crosstab(df.Gender, df.Team))
print("Displaying the distribution of teams for each gender.")
print(pd.crosstab(df.Team, df.Gender))













