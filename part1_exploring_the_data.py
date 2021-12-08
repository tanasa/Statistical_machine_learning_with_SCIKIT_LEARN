#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


# reading the data


# In[68]:


student_data = pd.read_csv("student-mat.csv", sep=';')
student_data.head(10)


# In[69]:


# dropping G1 and G2


# In[70]:


student_data.drop(['G1', 'G2'], axis=1, inplace=True)
student_data.head(10)


# In[71]:


# adding a COLUMN with the information regarding PASS and FAIL (the grade < 10)


# In[72]:


student_data['pass']=0
student_data.loc[student_data['G3'] >= 10, 'pass'] = 1 ## it updates the column "pass"
student_data.head(10)


# In[73]:


train_df['G3'].describe()


# In[74]:


train_df['pass'].describe()


# In[75]:


# defining the TRAIN and TEST datasets


# In[76]:


train_df, test_df = train_test_split(student_data, test_size=0.20, random_state=14)

print(train_df.shape)
print(test_df.shape)


# In[77]:


train_df.head(2)


# In[78]:


test_df.head(2)


# In[79]:


# saving the TRAINING and TESTING DATA in separate files


# In[80]:


train_df.to_csv('student-mat-train.csv', index=False)
test_df.to_csv('student-mat-test.csv', index=False)


# In[81]:


# Exploring the OUTPUT that is G3


# In[82]:


grade_counts = train_df['G3'].value_counts().sort_index()
grade_counts.plot(kind='bar') 


# In[83]:


# Filling in the MISSING VALUES
grade_counts = train_df['G3'].value_counts()
print(grade_counts)

for g in range(0,21): 
    if g not in grade_counts.index: # ie when g is 1,2,3,4
        grade_counts.loc[g] = 0


# In[84]:


grade_counts.sort_index(inplace=True)
print(grade_counts)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
grade_counts.plot(kind='bar', ax=ax) 


# In[85]:


# Exploring the DISTRIBUTION of PASS/FAIL


# In[87]:


fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
train_df['pass'].value_counts().sort_index().plot(kind='bar', ax=ax1)


# In[88]:


# Visualizing the distribution of the FEATURES


# In[89]:


# the feature "SCHOOL"


# In[90]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
train_df['school'].value_counts().sort_index().plot(ax=ax, kind="bar")
ax.set_title("school", fontsize=18)


# In[91]:


# the feature "SEX"


# In[92]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
train_df['sex'].value_counts().sort_index().plot(ax=ax, kind="bar")
ax.set_title("sex", fontsize=18)


# In[93]:


# the feature "AGE"


# In[94]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
train_df['age'].value_counts().sort_index().plot(ax=ax, kind="bar")
ax.set_title("age", fontsize=18)


# In[95]:


# the feature "FAMSIZE"


# In[96]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
train_df['famsize'].value_counts().sort_index().plot(ax=ax, kind="bar")
ax.set_title("famsize", fontsize=18)


# In[97]:


# the feature "FREETIME"


# In[98]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
train_df['freetime'].value_counts().sort_index().plot(ax=ax, kind="bar")
ax.set_title("freetime", fontsize=18)


# In[99]:


# the feature "HEALTH"


# In[100]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
train_df['health'].value_counts().sort_index().plot(ax=ax, kind="bar")
ax.set_title("health", fontsize=18)


# In[101]:


# the feature "ABSENCES"


# In[102]:


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
train_df['absences'].value_counts().sort_index().plot(ax=ax, kind="bar")
ax.set_title("absences", fontsize=18)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




