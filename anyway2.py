# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 18:38:32 2021

@author: LG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

titanic_df = pd.read_csv('titanic_train.csv')

# In[1]

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace = True)
titanic_df['Cabin'].fillna('N', inplace = True)
titanic_df['Embarked'].fillna('N',inplace = True)

titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
# In[2]
survived_sex = titanic_df.groupby(['Sex','Survived'])['Survived'].count()
print(survived_sex)

# In[3]
sns.barplot(x = 'Sex', y = 'Survived', data = titanic_df)
# In[4]
sns.barplot(x = 'Pclass',y = 'Survived', hue = 'Sex' ,data = titanic_df)
# In[5]
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12 : cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    
    return cat

plt.figure(figsize = (10,6))
group_names = ['Unknown', 'Baby','Child','Teenager','Student','Young Adult','Adult','Elderly']

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x = 'Age_cat',y = 'Survived',hue = 'Sex', data = titanic_df, order = group_names)
titanic_df.drop('Age_cat', axis = 1, inplace = True)
# In[6]
from sklearn import preprocessing
def encode_features(dataDF):
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
    
    return dataDF


titanic_df = encode_features(titanic_df)
# In[7]

