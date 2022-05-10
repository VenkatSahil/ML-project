#!/usr/bin/env python
# coding: utf-8

# # MACHINE LEARNING PROJECT USING LINEAR REGRESSION ON STEAM TABLES

# # Importing the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Importing the data

# In[2]:


data=pd.read_csv("steamtables_data.csv")


# # Original CSV data

# In[3]:


data


# # Splitting the data for testing and training

# In[4]:


split=np.random.rand(len(data))<0.8
train=data[split]
test=data[~split]
print("Training Data Shape",len(train))
print("Testing Data Shape",len(test))


# # Finding the Specific Volume of Saturated Liquid Using Linear Regression

# In[5]:


#Importing LinearRegression from Scikitlearn
from sklearn import linear_model
reg1=linear_model.LinearRegression()
#Taking the required data from the whole
inp1=np.asanyarray(train[['Temp','Sat_press']])
out1=np.asanyarray(train[['Spec_vol_Sat_liquid']])
reg1.fit(inp1,out1)
print("Reg_coeff",reg1.coef_)
print("Reg_intercept",reg1.intercept_)
#visualization of the original and predicted data
x = np.arange(0,len(train),1)
plt.scatter(x,out1,color = "blue")
plt.scatter(x,reg1.coef_[0][0]*train.Temp + reg1.coef_[0][1]*train.Sat_press+reg1.intercept_[0], color = "green")
plt.xlabel ("Temperature and Pressure")
plt.ylabel ("Specific Volume of Saturated Liquid")
plt.show()
inp1test = np.asanyarray(data[['Temp','Sat_press']])
out1test = np.asanyarray(data[['Spec_vol_Sat_liquid']])
out1pred = reg1.predict(inp1test)
#Finding the R2 score and Mean squared error for original and predicted outputs
from sklearn.metrics import mean_squared_error,r2_score
print ("MSE:",mean_squared_error (out1test,out1pred))
print ("R2 Score:",r2_score(out1test,out1pred))
print(reg1.predict(np.array([[356.99,18000]])))


# # Finding the Internal Energy of Saturated Liquid

# In[6]:


from sklearn import linear_model
reg2=linear_model.LinearRegression()
inp1=np.asanyarray(train[['Temp','Sat_press']])
out1=np.asanyarray(train[['Internal_energy_Sat_liquid']])
reg2.fit(inp1,out1)
print("Reg_coeff",reg2.coef_)
print("Reg_intercept",reg2.intercept_)
x = np.arange(0,len(train),1)
plt.scatter(x,out1,color = "blue")
plt.scatter(x,reg2.coef_[0][0]*train.Temp + reg2.coef_[0][1]*train.Sat_press+reg2.intercept_[0], color = "green")
plt.xlabel ("'Temperature and Pressure'")
plt.ylabel ("Internal_energy_Sat_liquid")
plt.show()
inp1test = np.asanyarray(data[['Temp','Sat_press']])
out1test = np.asanyarray(data[['Internal_energy_Sat_liquid']])
out1pred = reg2.predict(inp1test)
from sklearn.metrics import mean_squared_error,r2_score
print ("MSE:",mean_squared_error (out1test,out1pred))
print ("R2 Score:",r2_score(out1test,out1pred))
print(reg2.predict(np.array([[130.46,270.28]])))


# # Finding the Internal energy of saturated vapour

# In[7]:


from sklearn import linear_model
reg5=linear_model.LinearRegression()
inp1=np.asanyarray(train[['Temp','Sat_press']])
out1=np.asanyarray(train[['Internal_energy_Sat_vapour']])
reg5.fit(inp1,out1)
print("Reg_coeff",reg5.coef_)
print("Reg_intercept",reg5.intercept_)
x = np.arange(0,len(train),1)
plt.scatter(x,out1,color = "blue")
plt.scatter(x,reg5.coef_[0][0]*train.Temp + reg5.coef_[0][1]*train.Sat_press+reg5.intercept_[0], color = "green")
plt.xlabel ("'Temperature and Pressure'")
plt.ylabel ("Internal_energy_Evap")
plt.show()
inp1test = np.asanyarray(data[['Temp','Sat_press']])
out1test = np.asanyarray(data[['Internal_energy_Sat_vapour']])
out1pred = reg5.predict(inp1test)
from sklearn.metrics import mean_squared_error,r2_score
print ("MSE:",mean_squared_error (out1test,out1pred))
print ("R2 Score:",r2_score(out1test,out1pred))
print(reg5.predict(np.array([[230,2797.1]])))


# # Finding the Enthalpy of Saturated Liquid

# In[8]:


from sklearn import linear_model
reg6=linear_model.LinearRegression()
inp1=np.asanyarray(train[['Temp','Sat_press']])
out1=np.asanyarray(train[['Enthalpy_Sat_liquid']])
reg6.fit(inp1,out1)
print("Reg_coeff",reg6.coef_)
print("Reg_intercept",reg6.intercept_)
x = np.arange(0,len(train),1)
plt.scatter(x,out1,color = "blue")
plt.scatter(x,reg6.coef_[0][0]*train.Temp + reg6.coef_[0][1]*train.Sat_press+reg6.intercept_[0], color = "green")
plt.xlabel ("'Temperature and Pressure'")
plt.ylabel ("Enthalpy_Sat_liquid")
plt.show()
inp1test = np.asanyarray(data[['Temp','Sat_press']])
out1test = np.asanyarray(data[['Enthalpy_Sat_liquid']])
out1pred = reg6.predict(inp1test)
from sklearn.metrics import mean_squared_error,r2_score
print ("MSE:",mean_squared_error (out1test,out1pred))
print ("R2 Score:",r2_score(out1test,out1pred))
print(reg6.predict(np.array([[230,2797.1]])))


# # Finding the Enthalpy of Evaporation

# In[10]:


from sklearn import linear_model
reg7=linear_model.LinearRegression()
inp1=np.asanyarray(train[['Temp','Sat_press']])
out1=np.asanyarray(train[['Enthalpy_Evap']])
reg7.fit(inp1,out1)
print("Reg_coeff",reg7.coef_)
print("Reg_intercept",reg7.intercept_)
x = np.arange(0,len(train),1)
plt.scatter(x,out1,color = "blue")
plt.scatter(x,reg7.coef_[0][0]*train.Temp + reg7.coef_[0][1]*train.Sat_press+reg7.intercept_[0], color = "green")
plt.xlabel ("'Temperature and Pressure'")
plt.ylabel ("Enthalpy_Evap")
plt.show()
inp1test = np.asanyarray(data[['Temp','Sat_press']])
out1test = np.asanyarray(data[['Enthalpy_Evap']])
out1pred = reg7.predict(inp1test)
from sklearn.metrics import mean_squared_error,r2_score
print ("MSE:",mean_squared_error (out1test,out1pred))
print ("R2 Score:",r2_score(out1test,out1pred))
print(reg7.predict(np.array([[275.59,6000]])))


# # Finding the Enthalpy of Saturated Vapour

# In[11]:


from sklearn import linear_model
reg8=linear_model.LinearRegression()
inp1=np.asanyarray(train[['Temp','Sat_press']])
out1=np.asanyarray(train[['Enthalpy_Sat_vapour']])
reg8.fit(inp1,out1)
print("Reg_coeff",reg8.coef_)
print("Reg_intercept",reg8.intercept_)
x = np.arange(0,len(train),1)
plt.scatter(x,out1,color = "blue")
plt.scatter(x,reg8.coef_[0][0]*train.Temp + reg8.coef_[0][1]*train.Sat_press+reg8.intercept_[0], color = "green")
plt.xlabel ("'Temperature and Pressure'")
plt.ylabel ("Enthalpy_Sat_vapour")
plt.show()
inp1test = np.asanyarray(data[['Temp','Sat_press']])
out1test = np.asanyarray(data[['Enthalpy_Sat_vapour']])
out1pred = reg8.predict(inp1test)
from sklearn.metrics import mean_squared_error,r2_score
print ("MSE:",mean_squared_error (out1test,out1pred))
print ("R2 Score:",r2_score(out1test,out1pred))
print(reg8.predict(np.array([[275.59,6000]])))


# # Finding th Entropy of Saturated Liquid

# In[12]:


from sklearn import linear_model
reg9=linear_model.LinearRegression()
inp1=np.asanyarray(train[['Temp','Sat_press']])
out1=np.asanyarray(train[['Entropy_Sat_liquid']])
reg9.fit(inp1,out1)
print("Reg_coeff",reg9.coef_)
print("Reg_intercept",reg9.intercept_)
x = np.arange(0,len(train),1)
plt.scatter(x,out1,color = "blue")
plt.scatter(x,reg9.coef_[0][0]*train.Temp + reg9.coef_[0][1]*train.Sat_press+reg9.intercept_[0], color = "green")
plt.xlabel ("'Temperature and Pressure'")
plt.ylabel ("Entropy_Sat_liquid")
plt.show()
inp1test = np.asanyarray(data[['Temp','Sat_press']])
out1test = np.asanyarray(data[['Entropy_Sat_liquid']])
out1pred = reg9.predict(inp1test)
from sklearn.metrics import mean_squared_error,r2_score
print ("MSE:",mean_squared_error (out1test,out1pred))
print ("R2 Score:",r2_score(out1test,out1pred))
print(reg9.predict(np.array([[275.59,6000]])))


# # Finding the Entropy of Evaporation

# In[13]:


from sklearn import linear_model
reg10=linear_model.LinearRegression()
inp1=np.asanyarray(train[['Temp','Sat_press']])
out1=np.asanyarray(train[['Entropy_Evap']])
reg10.fit(inp1,out1)
print("Reg_coeff",reg10.coef_)
print("Reg_intercept",reg10.intercept_)
x = np.arange(0,len(train),1)
plt.scatter(x,out1,color = "blue")
plt.scatter(x,reg10.coef_[0][0]*train.Temp + reg10.coef_[0][1]*train.Sat_press+reg10.intercept_[0], color = "green")
plt.xlabel ("'Temperature and Pressure'")
plt.ylabel ("Entropy_Evap")
plt.show()
inp1test = np.asanyarray(data[['Temp','Sat_press']])
out1test = np.asanyarray(data[['Entropy_Evap']])
out1pred = reg10.predict(inp1test)
from sklearn.metrics import mean_squared_error,r2_score
print ("MSE:",mean_squared_error (out1test,out1pred))
print ("R2 Score:",r2_score(out1test,out1pred))
print(reg10.predict(np.array([[275.59,6000]])))


# # Finding the Entropy of Saturated Vapour

# In[14]:


from sklearn import linear_model
reg11=linear_model.LinearRegression()
inp1=np.asanyarray(train[['Temp','Sat_press']])
out1=np.asanyarray(train[['Entropy_Sat_vapour ']])
reg11.fit(inp1,out1)
print("Reg_coeff",reg11.coef_)
print("Reg_intercept",reg11.intercept_)
x = np.arange(0,len(train),1)
plt.scatter(x,out1,color = "blue")
plt.scatter(x,reg11.coef_[0][0]*train.Temp + reg11.coef_[0][1]*train.Sat_press+reg11.intercept_[0], color = "green")
plt.xlabel ("'Temperature and Pressure'")
plt.ylabel ("Entropy_Sat_vapour ")
plt.show()
inp1test = np.asanyarray(data[['Temp','Sat_press']])
out1test = np.asanyarray(data[['Entropy_Sat_vapour ']])
out1pred = reg11.predict(inp1test)
from sklearn.metrics import mean_squared_error,r2_score
print ("MSE:",mean_squared_error (out1test,out1pred))
print ("R2 Score:",r2_score(out1test,out1pred))
print(reg11.predict(np.array([[275.59,6000]])))


# In[ ]:




