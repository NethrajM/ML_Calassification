#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# In[26]:


Dataset=pd.read_csv("CKD-Calssification-Dataset.csv")


# In[27]:


Dataset


# In[28]:


Dataset=pd.get_dummies(Dataset,drop_first=True)


# In[29]:


Dataset


# In[30]:


Dataset.columns 


# In[31]:


independent=Dataset[['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hrmo', 'pcv',
       'wc', 'rc', 'sg_b', 'sg_c', 'sg_d', 'sg_e', 'rbc_normal', 'pc_normal',
       'pcc_present', 'ba_present', 'htn_yes', 'dm_yes', 'cad_yes',
       'appet_yes', 'pe_yes', 'ane_yes']]


# In[32]:


independent


# In[33]:


Dependent=Dataset[['classification_yes']]


# In[34]:


Dependent


# In[35]:


Dataset=Dataset.drop("age",axis=1)


# In[36]:


Dataset["classification_yes"].value_counts()


# In[37]:


independent = Dataset[[ 'bp', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hrmo', 'pcv', 'wc', 'rc', 'sg_b', 'sg_c', 'sg_d', 'sg_e', 'rbc_normal', 'pc_normal', 'pcc_present', 'ba_present', 'htn_yes', 'dm_yes', 'cad_yes', 'appet_yes', 'pe_yes', 'ane_yes']]

Dependent = Dataset['classification_yes']


# In[38]:


independent.shape


# In[39]:


Dependent


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(independent,Dependent,test_size=1/3,random_state=0)


# In[44]:


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)

classifier.fit(x_train, y_train)


# In[45]:


y_pred=Classifier.predict(x_test)


# In[46]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[47]:


print(cm)


# In[48]:


from sklearn.metrics import classification_report
clf_report=classification_report(y_test,y_pred)


# In[49]:


print(clf_report)


# In[50]:


print(cm)


# In[ ]:





# In[ ]:




