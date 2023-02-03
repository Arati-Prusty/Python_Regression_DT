#!/usr/bin/env python
# coding: utf-8

# In[ ]:


REGRESSION DECISION TREE(BOSTON DATA )


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


boston_data= pd.read_csv("D:\\python\\edureka\\DATASET\\additional_resources_6_xce_lwxlitn\\BostonHousing.csv")


# In[3]:


boston_data


# In[4]:


boston_data.describe()


# In[5]:


boston_data.info()


# In[6]:


boston_data.isnull()


# In[7]:


boston_data.isnull().sum()


# In[28]:


X= boston_data.iloc[:, 0:13].values  # before divide into train_test first covert them to array form


# In[29]:


X


# In[30]:


y= boston_data.iloc[:,13:14].values  


# In[31]:


y


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.3,random_state=121)


# In[35]:


x_train


# In[36]:


y_train


# In[42]:


from sklearn.tree import DecisionTreeRegressor


# In[43]:


regressor_dt= DecisionTreeRegressor()


# In[44]:


regressor_dt.fit(x_train,y_train)


# In[45]:


y_pred=regressor_dt.predict(x_test)


# In[46]:


y_pred


# In[47]:


from sklearn.metrics import mean_squared_error


# In[48]:


import math


# In[49]:


mse=mean_squared_error(y_pred,y_test)  # mean square error


# In[50]:


mse


# In[51]:


rmse=math.sqrt(mse)  # root mean square error


# In[52]:


rmse


# In[ ]:


#compare


# In[54]:


compare=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_pred)],axis=1)


# In[55]:


compare


# In[ ]:


# lower the root mean square (rmse) value,means there is very little difference
# between actual and predicted values. hence model is good


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




