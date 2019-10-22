#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ##############################################################
# #
# # Module import
# #some classifiers were not used either because they gave poor results or because i could not get them to work.
# ##############################################################
import pandas as pd
import numpy as np
import sklearn as skl
import time
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.7

import sklearn.linear_model as skl_lm
import category_encoders as ce
import seaborn as seabornInstance

from sklearn.datasets import fetch_mldata
from sklearn.utils import check_random_state
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# In[33]:


training_dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
test_dataset = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
training_dataset = training_dataset.dropna() #see below description of help needed from david farrell to have correct dimensions
training_dataset.shape


# In[ ]:


training_dataset.describe()


# In[3]:


dataset = pd.concat([training_dataset, test_dataset], sort=False)
dataset.shape


# In[4]:


dataset.dtypes


# In[5]:


#check which columns contain NAN values
dataset.isnull().any()


# In[6]:


#Cleaning age (try median too)
mean_age = dataset['Age'].median()
dataset ['Age'] = dataset['Age'].replace(np.nan, mean_age)
dataset ['Age'] = dataset['Age'].replace('unknown', mean_age)
#Cleaning year of record
mean_yearofrec = dataset['Year of Record'].median()
dataset['Year of Record'] = dataset['Year of Record'].replace(np.nan, mean_yearofrec)
#cleaning income (cant do this as dimensions of cells get very messy, better to just replace nan)
mean_income = dataset['Income in EUR'].median()
#dataset['Income in EUR'] = dataset['Income in EUR'].replace(np.nan, mean_income)


# In[7]:


#obj_dataset[obj_dataset.isnull().any(axis=1)]
dataset.isnull().any()


# In[8]:


dataset.shape


# In[9]:


X = dataset[['Instance', 'Year of Record', 'Gender', 'Age', 'Country', 'Size of City',
             'Profession', 'University Degree', 'Hair Color', 'Body Height [cm]']]

y = dataset['Income in EUR']


# In[10]:


#binary encoding better than one hot or .cat.codes
binary_encoder = ce.BinaryEncoder(cols=['Country', 'Profession','University Degree', 'Hair Color', 'Gender'])
binary_encoder.fit(X, y)
encoded_data = binary_encoder.transform(X)


# In[11]:


#CLEANING GENDER
#obj_dataset = obj_dataset[obj_dataset.Gender != "other"]
#obj_dataset = obj_dataset[obj_dataset.Gender != "0"]
#obj_dataset = obj_dataset[obj_dataset.Gender != "unknown"]
#CLEANING HAIR COLOR
#obj_dataset = obj_dataset[obj_dataset['Hair Color'] != "0"]
#obj_dataset = obj_dataset[obj_dataset['Hair Color'] != "Unknown"]
#CLEANING UNIVERSITY DEGREE
#obj_dataset = obj_dataset[obj_dataset['University Degree'] != "0"]


# In[12]:


#ONE HOT ENCODING FOR GENDER
#obj_dataset = pd.get_dummies(obj_dataset, columns=["Gender", "Hair Color", "University Degree"])


# In[ ]:


#NOTE: CELL HERE TO SPECIFIED below WAS WITH ASSISTANCE OF DAVID FARRELL AS I GOT VERY CONFUED
#WITH THE DIMENSIONS OF MY CELLS AND AFTER MANY HOURS ASKED FOR HELP


# In[13]:


X = encoded_data.reset_index()


# In[14]:


for i, data in X.iterrows():
    if data['Instance'] == 111994:
        break
    index = i
print(index)
X_train = X.iloc[:index+1, :]
X_test = X.iloc[index+1:, :]
print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[15]:


y.shape


# In[16]:


y = y.dropna()
y_train = np.array(y)
print(y_train.shape)


# In[ ]:


#END OF DAVID FARRELL HELP


# In[17]:


# obj_dataset["Profession"] = obj_dataset["Profession"].astype('category')
# obj_dataset["Country"] = obj_dataset["Country"].astype('category')
# obj_dataset.dtypes


# In[18]:


# obj_dataset["Profession_cat"] = obj_dataset["Profession"].cat.codes
# obj_dataset["Country_cat"] = obj_dataset["Country"].cat.codes
# obj_dataset.dtypes


# In[19]:


# obj_dataset = obj_dataset.drop("Country", axis=1)
# obj_dataset = obj_dataset.drop("Profession", axis=1)
# dataset = dataset.drop("Profession", axis = 1)
# dataset = dataset.drop("Gender", axis = 1)
# dataset = dataset.drop("University Degree", axis = 1)
# dataset = dataset.drop("Country", axis = 1)
# dataset = dataset.drop("Hair Color", axis = 1)
X_train = X_train.drop(['Instance', 'index'], axis=1)
X_test = X_test.drop(['Instance','index'], axis=1)
# dataset.dtypes


# In[20]:


# #dataset = pd.merge(left=dataset, right=obj_dataset)
# dataset = dataset.join(obj_dataset, how='inner')
# #income cant be int32
# dataset = dataset.astype({'Income in EUR': 'int64'})
# #dataset = dataset.astype({'Profession_cat': 'unit'})


# In[21]:


#Normalize dataset
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
#dataset = pd.DataFrame(x_scaled, columns = dataset.columns)


# In[22]:


plt.figure(figsize=(15,8))
plt.tight_layout()
seabornInstance.distplot(y_train)


# In[23]:


#Next, we split 80% of the data to the training set while 20% of the data to test set using below code.
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_train, y_train, test_size=0.8, random_state=0)


# In[24]:


#THIS WAS PREVIOUS ATTEMPT WITH LINEAR REGRESSION BUT RANDOM FOREST WORKED BETTER
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()  
#regressor.fit(X_train, y_train)
#y_pred = regressor.predict(X_test)


# In[25]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 500, n_jobs=-1, min_samples_split=2, min_samples_leaf=1, verbose=10, random_state = 0)
# Train the model on training data
rf.fit(X_train, y_train);


# In[26]:


# Use the forest's predict method on the test data
y_pred = rf.predict(X_test)
# Calculate the absolute errors
test_error = rf.predict(X_test_temp)
test_error_abs = abs(test_error - y_test_temp)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:','€', round(np.mean(test_error_abs), 2) )


# In[27]:


test_error.shape


# In[28]:


#DataFrame = y_test_temp
df = pd.DataFrame({'Actual':y_test_temp, 'Predicted': test_error})
df.shape


# In[29]:


df1 = df.head(50)


# In[30]:


#df.shape


# In[31]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[32]:


df = pd.read_csv('D:/Users/rocky/Desktop/Engineering/YEAR 5/CS7CS4 - Machine Learning/Comp 1/tcd ml 2019-20 income prediction submission file.csv')
print(df.shape)
df['Income'] = y_pred
df.to_csv('D:/Users/rocky/Desktop/Engineering/YEAR 5/CS7CS4 - Machine Learning/Comp 1/tcd ml 2019-20 income prediction submission file.csv', encoding = 'utf-8', index = False)


# In[ ]:




