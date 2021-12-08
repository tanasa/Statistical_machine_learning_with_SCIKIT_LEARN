#!/usr/bin/env python
# coding: utf-8

# In[1]:


# continuing the LOGISTIC REGRESSION by introducing REGULARIZATION
# in order to prevent OVER-FITTING


# In[2]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


np.random.seed(seed=1)


# In[ ]:





# In[ ]:


# reading the TRAINING DATA and using 1-HOT ENCODING


# In[4]:


train_df = pd.read_csv("student-mat-train.csv")
train_dummies_df = pd.get_dummies(train_df, drop_first=True)
train_dummies_df.head(3)


# In[ ]:


# reading the TESTING DATA and using 1-HOT ENCODING


# In[5]:


test_df = pd.read_csv("student-mat-test.csv")
test_dummies_df = pd.get_dummies(test_df, drop_first=True)
test_dummies_df.head(2)


# In[ ]:





# In[ ]:


# separate the data into TRAINING and TESTING datasets


# In[6]:


# separate our training data into predictors and predictand
xCols = [col for col in train_dummies_df.columns if col not in ['pass', 'G3']]
X_train = train_dummies_df[xCols]
y_train = train_dummies_df['pass']


# In[7]:


# separate our testing data into predictors and predictand
X_test = test_dummies_df[xCols]
y_test = test_dummies_df['pass']


# In[ ]:





# In[ ]:


# in order to prevent OVER-FITTING


# In[ ]:


# Feature filters are methods to simple reduce the number of features your model can train on. 

# Early-Stopping is the practice of stopping the training early as soon as the performance 
# in your cross-validation decreases

# Hyper-parameter settings 

# REGULARIZATION


# In[ ]:


# "LASSO" or "L1" pushes the coefficients to 0 and effectively removes the features completely. 
# "Ridge" or L2" does not reduce coefficients all the way to zero.


# In[9]:


# to apply REGULARIZATION
# to change the C value when we create our LogisticRegression object. 
# the smaller the value of C, the stronger the regularization. 


# In[10]:


reg_params = [1e15, 10000, 1000, 100, 10, 1, 0.1, 0.01]
reg_params = np.arange(0.5, 0.01, -0.01)

train_scores = []

test_scores = []

coefs = []

for c in reg_params:
    # Create and fit a model with the value of c
    logReg = LogisticRegression(C=c, penalty='l2')
    logReg.fit(X_train, y_train)
    coefs.append(logReg.coef_.reshape(-1))
    
    # Find the accuracy on the training data
    train_data_accuracy = accuracy_score(logReg.predict(X_train), y_train)
    train_scores.append(train_data_accuracy)
    
    # Find the accuracy on the testing data
    test_data_accuracy = accuracy_score(logReg.predict(X_test) , y_test)
    test_scores.append(test_data_accuracy)


# In[13]:


fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(reg_params, train_scores, label='train scores', marker='o')
ax.plot(reg_params, test_scores, label='test scores', marker='o')

ax.legend()
ax.set_title('Accuracy on Training and Testing Data With Respect To Regularization')
ax.set_xlabel('C Value')
ax.set_ylabel('Accuracy')


# In[ ]:





# In[ ]:


# VALIDATION SET
# A VALIDATION SET is taken from the TRAINING SET and used to set HYPER-PARAMETER values (like REGULARIZATION). 


# In[ ]:


# CROSS VALIDATION is when you iterate through different VALIDATION SETS in the TRAINING DATA.


# In[ ]:





# In[ ]:


# applying CROSS-VALIDATED REGULARIZATION


# In[15]:


reg_params = [1e15, 10000, 1000, 100, 10, 1, 0.1, 0.01]
reg_params = np.arange(0.5, 0.01, -0.005)

logRegCV = LogisticRegressionCV(Cs=reg_params, penalty='l1', cv=5, solver="liblinear")

# it loops through the values of reg_params using a cross_vaidation = 5
# for each value of C, it does the CV CROSS VALIDATION

logRegCV.fit(X_train, y_train)


# In[ ]:


# print(logRegCV.scores_)
# print(logRegCV.scores_[1].mean(axis=0))
# print(logRegCV.C_)


# In[ ]:


# The scores_ attribute gives us the scores, per C value, per cross-validation fold, for each class. 
# We can average across cross validation folds to find the best value of C in terms of accuracy.
# The C_ gives the best value of C.


# In[16]:


fig, axs = plt.subplots(1, 1, figsize=(14, 7))
axs.plot(logRegCV.Cs_, logRegCV.scores_[1].mean(axis=0), marker='o')
axs.set_xscale('log')
axs.set_title('')
axs.set_xlabel('C Values')
axs.set_ylabel('Accuracy')
axs.grid()

# printing max accuracy
print(logRegCV.scores_[1].mean(axis=0).max())


# In[ ]:





# In[17]:


# SOME COEFFICIENTS have been pushed to ZERO !
print(logRegCV.coef_)


# In[18]:


# PRINTING out the columns that correspond to the non-zero coefficients
for coef, col in zip(logRegCV.coef_[0], X_train.columns):
    if coef != 0:
        print('{:.3f} * {}'.format(coef, col))


# In[19]:


# the FEATURES that have been retained by the MODEL
print('We have reduced from {} features to {}'.format(X_train.shape[1], sum(logRegCV.coef_[0] != 0)))


# In[20]:


X_train.describe()


# In[ ]:





# In[ ]:


# DATA STANDARDIZATION


# In[ ]:


# The training algorithms can more effectively find the optimum when the input features are mean standardized. 
# We adjust the features so they have a mean of 0 and a standard deviation of 1. 
# i.e. we substract the mean from each feature, and then we divide by the standard deviation.


# In[22]:


X_train_standardized = (X_train - X_train.mean()) / X_train.std()
X_train_standardized.describe()


# In[23]:


X_test_standardized = (X_test - X_train.mean()) / X_train.std()
X_test_standardized.describe()


# In[ ]:





# In[ ]:


### using LOGISTIC REGRESSION on STANDARDIZED DATA


# In[24]:


logRegCVSD = LogisticRegressionCV(Cs=300, penalty='l1', cv=5, solver="liblinear", random_state=0)
logRegCVSD.fit(X_train_standardized, y_train)


# In[27]:


# the display :
fig, axs = plt.subplots(1, 1, figsize=(14, 7))
axs.plot(logRegCVSD.Cs_, logRegCVSD.scores_[1].mean(axis=0), marker='o')
axs.set_xscale('log')
axs.set_title('')
axs.set_xlabel('C Values')
axs.set_ylabel('Accuracy')
axs.grid()


# In[28]:


# printing the max accuracy
print(logRegCVSD.scores_[1].mean(axis=0).max())
print(logRegCVSD.C_)


# In[29]:


# print the coefficients!
print(logRegCVSD.coef_)


# In[30]:


# Now print out the columns that correspond to the non-zero coefficients
for coef, col in zip(logRegCVSD.coef_[0], X_train.columns):
    if coef != 0:
        print('{:.3f} * {}'.format(coef, col))


# In[31]:


# How many features have we retained in the model?
print('We have reduced from {} features to {}'.
      format(X_train_standardized.shape[1], sum(logRegCVSD.coef_[0] != 0)))


# In[ ]:





# In[ ]:


# COMPUTING the ACCURACY of these MODELS


# In[32]:


logRegAccuracy = accuracy_score(logRegCV.predict(X_test), y_test)
logRegSDAccuracy = accuracy_score(logRegCVSD.predict(X_test_standardized), y_test)


# In[33]:


print('logRegCV has an accuracy {:.2f}% on the test set.'.format(logRegAccuracy*100))
print('logRegCVSD has an accuracy {:.2f}% on the test set.'.format(logRegSDAccuracy*100))


# In[ ]:





# In[ ]:


# COMPARING the PREDICTIONS


# In[35]:


print(logRegCV.predict(X_test))
print(logRegCVSD.predict(X_test_standardized))


# In[ ]:





# In[39]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, logRegCV.predict(X_test))
cnf_matrix


# In[40]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, logRegCVSD.predict(X_test_standardized))
cnf_matrix


# In[ ]:





# In[ ]:


# the CLASSIFICATION REPORT


# In[ ]:





# In[42]:


# PRECISION : This gives the percentage of predictions that the model made for the class that were correct. 
# The formula for this is TP/(TP + FP) for the positive class and TN/(TN + FN) for the negative class.


# In[ ]:


# RECALL : This gives the percentage of the class, in the data set, that were correctly labeled by the model. 
# The formula for this is TP/(TP + FN) for the positive class and TN/(TN + FP) for the negative class.


# In[43]:


# F1-SCORE : This is like a weighted average of the precision and recall and can be used to compare the performance on different classes.


# In[ ]:





# In[44]:


print(classification_report(y_test, logRegCV.predict(X_test)))


# In[45]:


print(classification_report(y_test, logRegCV.predict(X_test_standardized)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




