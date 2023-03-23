#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# In[2]:


Dataset=pd.read_csv("CKD-Calssification-Dataset.csv")


# In[3]:


Dataset


# In[4]:


Dataset=pd.get_dummies(Dataset,drop_first=True)


# In[5]:


Dataset


# In[6]:


Dataset.columns 


# In[7]:


independent=Dataset[['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hrmo', 'pcv',
       'wc', 'rc', 'sg_b', 'sg_c', 'sg_d', 'sg_e', 'rbc_normal', 'pc_normal',
       'pcc_present', 'ba_present', 'htn_yes', 'dm_yes', 'cad_yes',
       'appet_yes', 'pe_yes', 'ane_yes']]


# In[8]:


independent


# In[9]:


Dependent=Dataset[['classification_yes']]


# In[10]:


Dependent


# In[11]:


Dataset=Dataset.drop("age",axis=1)


# In[12]:


Dataset["classification_yes"].value_counts()


# In[13]:


independent = Dataset[[ 'bp', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hrmo', 'pcv', 'wc', 'rc', 'sg_b', 'sg_c', 'sg_d', 'sg_e', 'rbc_normal', 'pc_normal', 'pcc_present', 'ba_present', 'htn_yes', 'dm_yes', 'cad_yes', 'appet_yes', 'pe_yes', 'ane_yes']]

Dependent = Dataset['classification_yes']


# In[14]:


independent.shape


# In[15]:


Dependent


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(independent,Dependent,test_size=1/3,random_state=0)


# In[20]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report 

classifier = GaussianNB()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

clf_report = classification_report(y_test, y_pred)

print(clf_report)

print(cm)


# In[23]:



from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report 

classifier = MultinomialNB()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

clf_report = classification_report(y_test, y_pred)

print(clf_report)

print(cm)






# In[29]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report 

classifier = BernoulliNB()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

clf_report = classification_report(y_test, y_pred)

print(clf_report)

print(cm)






# In[32]:


from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, classification_report 

classifier = CategoricalNB()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

clf_report = classification_report(y_test, y_pred)

print(clf_report)

print(cm)






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




