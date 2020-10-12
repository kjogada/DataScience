#Data Preprocessing

#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into training set ad test set

"""from sklearn.model_selection import train_test_split
X_train, X_Test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
#Feature scaling
"""
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_Test = standardscaler.transform(X_Test)"""


#Fitting Regression model to data set

#Prediction of result


#Visualising the Polynomiallinear regression model

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Polynomial Linear Regression')
plt.xlabel('Position levels')
plt.ylabel('Salaries')
plt.show()

#Visualising the Polynomiallinear regression model(For higher resolution)
X_grid = np.arrange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len()X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Polynomial Linear Regression')
plt.xlabel('Position levels')
plt.ylabel('Salaries')
plt.show()