#Data Preprocessing

#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into training set ad test set

from sklearn.model_selection import train_test_split
X_train, X_Test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Feature scaling
"""
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_Test = standardscaler.transform(X_Test)"""

#Fitting simple linear regression to training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test results

y_pred = regressor.predict(X_Test)

#Visualization 

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_Test, y_pred, color= 'blue' )
plt.scatter(X_Test, y_test, color = 'green')
behavior in Preferences > Help.
plt.show()