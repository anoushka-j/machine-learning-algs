#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:38:01 2023

@author: anoushkajawale
"""

import numpy as np

#w/o adding the dtype its int64, which is really too big
# for our numbers 

#making arrays
a = np.array([1, 2, 3], dtype="int16")
print(a)
b = np.array([4, 5, 6])
c = a + b
d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(c)

#get dimension of array 
print(c.ndim)

#get shape/description in rows and columns of arr
print(d.shape)

#get datatype 
print(a.dtype) #int16
print(d.dtype) #int64 cuz not specified 

#get byte size 
print(a.itemsize) #2 because its int16, but 1 if int8
print(d.itemsize) #8 because its int64 since not specified dtype

#get num elements 
print(a.size) #3 
print(d.size) #6

#get byte size - same thing as above 
print(a.nbytes) # 6 = itemsize * size = 2 * 3 elements of list
print(d.nbytes) # 48 = itemsize * size = 8 * 6 elements of list


#accessing/indexing in np.arrays 
# : is for accessing along the same axis (row or column)
# , is for separating the rows and columns 
#rows
print(d[1, 2]) #first element, second index (which is really the third)
print(a[2]) #a is of ndim = 1, so can't use a double index 
print(d[0:1]) #first element, from 0 to 1
print(d[1:2]) #second element, from 0 to 1
print(d[1:]) #everything onwards from and including the first element

#columns
print(d[:, 0]) #get first column 
print(d[:, 1]) #second 


#changing elements 
d[0,1] = 100
print(d)
d[2, 2] = 200


#creating an all zeroes matrix
x = np.ones((2, 3, 4))

#matrix of random decimal numbers
y = np.random.rand(4, 2, 3)

#generate random integer value
z = np.random.randint(1, 9)

#generate array of random integers 
z1 = np.random.randint(1, 10, size=(4, 3))

"""identity matrix - when doing matrix multiplication 
with another matrix of compatible size, the result should be 
the original matrix."""
 
i = np.identity(3)
i_test = np.random.randint(1, 10, size=(3, 3))

#matrix multiplication - either one works
result = i @ i_test
result1 = np.matmul(i, i_test)

#checking if chatgpt is lying about the orig and result being equal 
#:) 
if np.array_equal(i_test, result) : 
    print(True)

#arithmetic on arrays 
a1 = np.array([1, 2, 3, 4])
a2 = a1 + 3
a2 = a1 * 12

#boolean masking in numpys 
#each element is True or False depending on a certain condition 

test1 = np.random.randint(10, 50, size=(5, 7))
boolean_mask = test1 > 30

a = np.array([[1,2,3,4,5],
              [4,5,6,7,8],
              [9,8,7,6,5]]) 

a = np.array([1, 2, 3])
b = a.view()
b[0] = 100

