#!/usr/bin/env python
# coding: utf-8

# In[19]:


#Install the dependencies
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math


# In[20]:


# load the dataset
raw = pd.read_csv('C:/Users/User/Downloads/bbri.csv', sep=',', header=0, index_col=0, engine='python', usecols=[0, 4], parse_dates=True)
dataframe = pd.read_csv('C:/Users/User/Downloads/bbri.csv', sep=',', usecols=[4], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float')
print(raw)


# In[3]:


# A variable for predicting 'n' days out into the future
forecast_out = 30 #'n=30' days
#Create another column (the target ) shifted 'n' units up
raw['Prediction'] = raw[['Close']].shift(-forecast_out)
#print the new data set
print(raw.tail())


# In[4]:


### Create the independent data set (X)  #######
# Convert the dataframe to a numpy array
X = np.array(raw.drop(['Prediction'],1))

#Remove the last '30' rows
X = X[:-forecast_out]
print(X)


# In[5]:


### Create the dependent data set (y)  #####
# Convert the dataframe to a numpy array 
y = np.array(raw['Prediction'])
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]
print(y)


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[7]:


# Create and train the Support Vector Machine (Regressor) 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(x_train, y_train)


# In[8]:


# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)


# In[11]:


# Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(raw.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)


# In[12]:


# Print support vector regressor model predictions for the next '30' days
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)


# In[22]:


# menampilkan grafik
plt.plot(dataset)
plt.plot(svm_prediction)


# In[ ]:




