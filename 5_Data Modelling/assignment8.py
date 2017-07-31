import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
os.chdir("Datasets")
from sklearn import linear_model
matplotlib.style.use('ggplot') # Look Pretty


def drawLine(model, X_test, y_test, title):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_test, y_test, c='g', marker='o')
  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

  print "Est 2014 " + title + " Life Expectancy: ", model.predict([[2014]])[0]
  print "Est 2030 " + title + " Life Expectancy: ", model.predict([[2030]])[0]
  print "Est 2045 " + title + " Life Expectancy: ", model.predict([[2045]])[0]

  score = model.score(X_test, y_test)
  title += " R2: " + str(score)
  ax.set_title(title)

  plt.show()

#
# Loading up the dataset 
X = pd.read_csv("life_expectancy.csv", sep='\t', header = 0)
print(X.describe())
print(X)

#
# Creating a linear regression model.
model = linear_model.LinearRegression()

#
# Slicing out the where Year is less than 1986 for WhiteMale
X_train = X["Year"][X["Year"]<1986]
y_train = X["WhiteMale"][X["Year"]<1986]

X_train = X_train.reshape(-1,1)
print(X_train.shape) 
print(y_train.shape)

#
# Training the model 
model.fit(X_train, y_train)
drawLine(model, X_train, y_train, "WhiteMale")



# Printing the actual 2014 WhiteMale life expectancy to compare with predicted value
# 
print(X["WhiteMale"][X["Year"]==2014])


#
# Slicing out the where Year is less than 1986 for BlackFemale
X_train = X["Year"][X["Year"]<1986]
y_train = X["BlackFemale"][X["Year"]<1986]

X_train = X_train.reshape(-1,1)

model.fit(X_train, y_train)
drawLine(model, X_train, y_train, "BlackFemale")

print(X["BlackFemale"][X["Year"]==2014])


#
# Printing out a correlation matrix
plt.imshow(X.corr(), cmap = plt.cm.Blues, interpolation = 'nearest')
plt.colorbar()
tick_marks = [i for i in range(len(X.columns))]
plt.xticks(tick_marks, X.columns, rotation = 'vertical')
plt.yticks(tick_marks, X.columns)

plt.show()
