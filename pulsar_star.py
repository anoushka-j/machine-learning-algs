#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 08:22:46 2023

@author: anoushkajawale
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer



#function to train model and print accuracy and classification report 
def score_method_log_reg(x_train, y_train, x_test, y_test) : 
    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    #results['class'] = y_pred
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy {accuracy: }')
    print("Classification Report: \n", classification_report(y_test, y_pred))
    print(f'MSE log reg: {mean_absolute_error(y_test, y_pred)}')
    return y_pred, probs

def score_method_rand_forest(x_train, y_train, x_test, y_test) : 
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'MSE rand forest: {mean_absolute_error(y_test, y_pred)}')
    return y_pred 
    
#SIMPLE LOGISTIC REGRESSION, DROPPING NAN COLS
trained_data = pd.read_csv("pulsar_data_train.csv")
x = trained_data.drop("target_class", axis=1)
dropped_cols_x = x.dropna(axis=1)
y = trained_data["target_class"]

x_train, x_test, y_train, y_test = train_test_split(dropped_cols_x, y, test_size=0.3, random_state=42)
print(score_method_log_reg(x_train, y_train, x_test, y_test))
print(score_method_rand_forest(x_train, y_train, x_test, y_test))
result_dropped_col_log_reg = pd.DataFrame(x_test)
preds, probs = score_method_log_reg(x_train, y_train, x_test, y_test)
result_dropped_col_log_reg['preds'] = preds
probs = np.array(probs)
result_dropped_col_log_reg['prob_0'] = probs[:, 0]
result_dropped_col_log_reg['prob_1'] = probs[:, 1]
result_dropped_col_log_reg.reset_index(drop=True, inplace=True)
result_dropped_col_log_reg.to_csv("log_reg_results.csv")
#results = np.array(results)
o_probs = probs[:, 0]
y_probs = [0 if prob >= 0.5 else 1 for prob in o_probs]
print(f'ROC AUC score: {roc_auc_score(y_test, y_probs)}')
print(f'f1 score: {f1_score(y_test, preds)}')

#SIMPLE LOGISTIC REGRESSION, DATA IMPUTATION
#finding the columns with missing values 
nan_sum_cols = x.isnull().sum()
#print(nan_sum_cols[nan_sum_cols > 0])
nan_sum_cols = pd.DataFrame(nan_sum_cols)
#print(nan_sum_cols)
nan_cols = [col for col in x_train.columns if x_train[col].isnull().any()]
#print(nan_cols)

trained_data = pd.read_csv("pulsar_data_train.csv")
x = trained_data.drop("target_class", axis=1)
dropped_cols_x = x.dropna(axis=1)
y = trained_data["target_class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

imputer = SimpleImputer(strategy='mean')
imputed_x_train = pd.DataFrame(imputer.fit_transform(x_train))
imputed_x_train.columns = x_train.columns 
imputed_x_test = pd.DataFrame(imputer.transform(x_test))
imputed_x_test.columns = x_test.columns
print(score_method_rand_forest(imputed_x_train, y_train, imputed_x_test, y_test))


#print("Results: imputation and logistic regression provided the least MSE, rather than any combination with dropping columns and random forest regressor.")








