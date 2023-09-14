#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 08:22:46 2023

@author: anoushkajawale
"""

import pandas as pd
import numpy as np


data = np.array(pd.read_csv("customer_data.csv"))
cols = pd.read_csv("customer_data.csv")
print(cols.columns)