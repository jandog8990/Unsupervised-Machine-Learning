#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:46:03 2020

@author: alejandrogonzales
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd

#Even spaced nums
x = np.linspace(start=-10, stop=10, num=1000)
#Normal cont RV
# Location (loc) key is mean. Scale (scale) ey is std dev
y = stats.norm.pdf(x, loc=0, scale=1.5)
plt.plot(x, y)
plt.show()

# read test dataset
df = pd.read_csv("bimodal_example.csv")
print(df.head(n=5))     # 5 users amount used in bitcoin (single feature)

# histogram of distribution
data = df.x
sns.distplot(data, bins=20, kde=False) # bins limits the samples
