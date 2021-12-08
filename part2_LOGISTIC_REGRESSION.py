#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# building the MODEL based on LOGISTIC REGRESSION


# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


train_df = pd.read_csv('student-mat-train.csv')
train_df.head(5)


# In[ ]:





# In[8]:


# CONVERTING CATEGORICAL DATA to NUMERICAL VARIABLES : ONE HOT ENCODING
# using the function :  pd.get_dummies()


# In[9]:


internet_dummies = pd.get_dummies(train_df[['internet']], drop_first=True)
internet_dummies.head()


# In[10]:


internet_dummies = pd.get_dummies(train_df[['Mjob']], drop_first=True)
internet_dummies.head()


# In[11]:


internet_dummies = pd.get_dummies(train_df[['Fjob']], drop_first=True)
internet_dummies.head()


# In[13]:


internet_dummies = pd.get_dummies(train_df[['romantic']], drop_first=True)
internet_dummies.head()


# In[ ]:





# In[14]:


# APPLYING ONE-HOT ENCODING to the ENTIRE TRAINING DATAFRAME


# In[ ]:





# In[15]:


train_dummies_df = pd.get_dummies(train_df, drop_first=True)
train_dummies_df.head()


# In[18]:


train_dummies_df.columns.sort_values()


# In[ ]:





# In[ ]:





# In[ ]:


# APPLYING ONE-HOT ENCODING to the ENTIRE TESTING DATAFRAME


# In[19]:


test_df = pd.read_csv('student-mat-test.csv')
test_df.head(2)


# In[21]:


test_dummies_df = pd.get_dummies(test_df, drop_first=True)
test_dummies_df.head(2)


# In[ ]:





# In[23]:


print(test_dummies_df.shape)
print(train_dummies_df.shape)
# print(train_dummies_df.columns)
# print(test_dummies_df.columns)


# In[ ]:





# In[ ]:


# making the PREDICTORS and the PREDICTAND (OUTPUT) variable


# In[31]:


# separate the TRAINING DATA into predictors and predictand
xCols = [col for col in train_dummies_df.columns if col not in ['pass', 'G3']]
X_train = train_dummies_df[xCols]
y_train = train_dummies_df['pass']
X_train.head()
y_train.head()


# In[32]:


# separate the TESTING DATA into predictors and predictand
X_test = test_dummies_df[xCols]
y_test = test_dummies_df['pass']
X_test.head()
y_test.head()


# In[ ]:





# In[33]:


# TRAINING the LOGISTIC REGRESSION MODEL


# In[34]:


## making a logistic regression OBJECT 
logReg = LogisticRegression(C=1e15)
logReg

## to train it, and set the values of the coefficients
logReg.fit(X_train, y_train)


# In[ ]:


# INSPECTING the TRAINING MODEL


# In[36]:


print(logReg.intercept_)


# In[37]:


print(logReg.coef_)


# In[38]:


print(X_train.columns)   


# In[ ]:


# PRINTING the EQUATION


# In[39]:


equation = '{:.2f}'.format(logReg.intercept_[0])
for (coef, feature) in zip(logReg.coef_[0], X_train.columns):
    equation += ' + {:.2f}({})'.format(coef, feature)
print(equation)


# In[40]:


z = logReg.intercept_ + np.dot(logReg.coef_, X_train.iloc[[0], :].values.T)
1 / (1+np.exp(-z))


# In[ ]:





# In[42]:


# COMPUTING the PROBABILITY for BOTH CLASSES
# it will give you the probability predicted for both clases (these will sum to 1)
# by using .predict_proba() method.


# In[43]:


# predicts the probability of each class (failing, passing)
print(logReg.predict_proba(X_train.iloc[[1], :]))


# In[ ]:


# and for the entire DATASET


# In[46]:


# for each student, it predict PROBAB_FAILING, PROBAB_PASSING
logReg.predict_proba(X_train)


# In[ ]:





# In[48]:


# COMPUTING the ACCURACY of the MODEL on TRAINING DATA 


# In[51]:


pred_train = logReg.predict(X_train)
pred_train
train_data_accuracy  = accuracy_score(pred_train, y_train)
train_data_accuracy


# In[47]:


# COMPUTING the ACCURACY of the MODEL on TESTING DATA 


# In[52]:


pred_test = logReg.predict(X_test)
pred_test
test_data_accuracy  = accuracy_score(pred_test, y_test)
test_data_accuracy


# In[ ]:


# Printing the ACCURACY


# In[54]:


## printing 
print("Accuracy on the training data {:.2f}%".format(train_data_accuracy * 100))
print("Accuracy on the test data {:.2f}%".format(test_data_accuracy * 100))


# In[ ]:





# In[ ]:


# It looks that the MODEL OVERFITS the DATA


# In[ ]:





# In[ ]:




