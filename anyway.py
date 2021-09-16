# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:37:55 2021

@author: LG
"""

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns = iris.feature_names)

# In[1]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)


iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)

print(iris_df_scaled.mean())

# In[2]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)

# In[3]