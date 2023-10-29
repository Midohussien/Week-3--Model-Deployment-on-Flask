#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[4]:


HousePrice = pd.read_csv('Housing.csv')


# In[5]:


HousePrice.keys()


# In[9]:


x = HousePrice[[ 'area', 'bedrooms', 'bathrooms', 'stories',
       'parking']]
y = HousePrice['price']


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=101)


# In[11]:


model= LinearRegression()
model.fit(x_train,y_train)


# In[ ]:


pickle.dump(model,open('model.pickle','wb'))

