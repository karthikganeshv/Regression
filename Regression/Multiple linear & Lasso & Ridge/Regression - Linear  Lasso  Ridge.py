#!/usr/bin/env python
# coding: utf-8

# In[151]:


import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[152]:


data = pd.read_csv('Downloads//CarPrice_Assignment.csv')
data.head()


# In[153]:


data.shape


# In[154]:


data.dropna(inplace=True)


# In[155]:


data.isnull().sum()


# In[156]:


data.info()


# In[157]:


data.drop(['CarName','car_ID'] , inplace = True, axis = 1)
data.shape


# In[158]:


import matplotlib.pyplot as plt 
import seaborn as sns 


# In[159]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr() , cmap="YlGnBu", annot=True)


# In[160]:


data = pd.get_dummies(data, drop_first =True)
data.head()


# In[161]:


data.shape


# In[162]:


data.columns


# In[163]:


data.corr()


# In[165]:


X = data.drop('price' , axis = 1)
y = data['price']


# In[167]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( X,y , test_size = 0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)


# In[168]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)


# In[169]:


reg.score(X_train, y_train)


# In[171]:


y_pred = reg.predict(X_test)


# In[172]:


from sklearn.metrics import mean_squared_error

MSE  = mean_squared_error(y_test, y_pred)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)


# In[173]:


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("R2 :" ,r2)


# # Lasso Regression

# In[174]:


#Lasso Regression
from sklearn import linear_model
lasso  = linear_model.Lasso(alpha=1 , max_iter= 3000)

lasso.fit(X_train, y_train)


# In[175]:


lasso.score(X_train, y_train)


# In[176]:


y_pred_l = lasso.predict(X_test)


# In[177]:


MSE  = mean_squared_error(y_test, y_pred_l)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)

r2 = r2_score(y_test, y_pred_l)
print("R2 :" ,r2)


# # Rigde regression

# In[178]:


from sklearn.linear_model import Ridge

ridge  = Ridge(alpha=0.1)


# In[179]:


ridge.fit(X_train,y_train)


# In[180]:


ridge.score(X_train, y_train)


# In[181]:


y_pred_r = ridge.predict(X_test)


# In[182]:


MSE  = mean_squared_error(y_test, y_pred_r)
print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
print("RMSE :" ,RMSE)

r2 = r2_score(y_test, y_pred_r)
print("R2 :" ,r2)


# In[ ]:




