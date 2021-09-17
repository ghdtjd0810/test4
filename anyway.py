# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 19:10:21 2021

@author: LG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

# In[1]

def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Cabin'].fillna('N', inplace = True)
    df['Embarked'].fillna('N', inplace = True)
    df['Fare'].fillna(0, inplace = True)
    
    return df

def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'], axis = 1, inplace = True)
    return df

def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
        
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

titanic_df = pd.read_csv('titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis = 1) #it seems like target

# In[2]
X_titanic_df = transform_features(X_titanic_df)

# In[3]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df,
                                                    test_size = 0.2, random_state = 11)

# In[4]
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# In[5]

dt_clf = DecisionTreeClassifier(random_state = 11)
rf_clf = RandomForestClassifier(random_state = 11)
lr_clf = LogisticRegression()


# In[6] -Decision


dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)

rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)


lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)

# In[7]
print(accuracy_score(y_test, dt_pred), accuracy_score(y_test, rf_pred),accuracy_score(y_test, lr_pred))

# In[8]

from sklearn.model_selection import KFold

def exec_kfold(clf, folds = 5):
    kfold = KFold(n_splits = folds)
    
    scores = []
    
    for iter_count,(train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        
        X_train, X_test =  X_titanic_df.values[train_index],X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        
    
    mean_score = np.mean(scores)
    print(mean_score)

exec_kfold(dt_clf, folds = 5)

# In[9]

