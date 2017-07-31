import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.chdir("Datasets")
from sklearn import linear_model
from sklearn.model_selection import train_test_split

matplotlib.style.use('ggplot') # Look Pretty


def drawLine(model, X_test, y_test, title, R2):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_test, y_test, c='g', marker='o')
  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

  title += " R2: " + str(R2)
  ax.set_title(title)
  print title
  print "Intercept(s): ", model.intercept_

  plt.show()

def drawPlane(model, X_test, y_test, title, R2):
  # Plotting test observations and comparing them to the regression plane,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_zlabel('prediction')


  X_test = np.array(X_test)
  col1 = X_test[:,0]
  col2 = X_test[:,1]

  # Set up a Grid. We could have predicted on the actual
  # col1, col2 values directly; but that would have generated
  # a mesh with WAY too fine a grid, which would have detracted
  # from the visualization
  x_min, x_max = col1.min(), col1.max()
  y_min, y_max = col2.min(), col2.max()
  x = np.arange(x_min, x_max, (x_max-x_min) / 10)
  y = np.arange(y_min, y_max, (y_max-y_min) / 10)
  x, y = np.meshgrid(x, y)

  # Predict based on possible input values that span the domain
  # of the x and y inputs:
  z = model.predict(  np.c_[x.ravel(), y.ravel()]  )
  z = z.reshape(x.shape)

  ax.scatter(col1, col2, y_test, c='g', marker='o')
  ax.plot_wireframe(x, y, z, color='orange', alpha=0.7)
  
  title += " R2: " + str(R2)
  ax.set_title(title)
  print title
  print "Intercept(s): ", model.intercept_
  
  plt.show()
  


X = pd.read_csv("College.csv", sep=',', header = 0, index_col = 0)
print(X.dtypes)

X.Private = X.Private.map({'Yes':1, 'No':0})


#
# Creating a linear regression model 
model = linear_model.LinearRegression()


 
### Number of accepted students as a function of the amount charged for room and board

#
# Creating two slices to store the room and board column, and the accepted
# students column. 
X_train = X[['Room.Board']]
y_train = X.Accept

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = .3, random_state = 7)


#
# Fitting the training data and scoring the model 
model.fit(X_train, y_train)
model.predict(X_test)
score = model.score(X_test,y_test)
drawLine(model, X_test, y_test, "Accept(Room&Board)", score)




### Number of accepted students as a function of the number of enrolled students per college.

model = linear_model.LinearRegression()
X_train = X[['Enroll']]
y_train = X.Accept

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = .3, random_state = 7)

#
# Fitting the training data and scoring the model
model.fit(X_train, y_train)
model.predict(X_test)
score = model.score(X_test,y_test)
drawLine(model, X_test, y_test, "Accept(Enroll)", score)



### Number of accepted students as as function of the numbr of failed undergraduate students per college.

model = linear_model.LinearRegression()
X_train = X[['F.Undergrad']]
y_train = X["Accept"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = .3, random_state = 7)

#
# Fitting the training data and scoring the model
model.fit(X_train, y_train)
model.predict(X_test)
score = model.score(X_test,y_test)
drawLine(model, X_test, y_test, "Accept(F.Undergrad)", score)


### Multivariate Linear Regression
### The amount charged for room and board AND the number of enrolled students as a
### function of the number of accepted students. 

model = linear_model.LinearRegression()

# Creating a slice that contains both the needed columns
X_train = X[['Room.Board','Enroll']]
y_train = X.Accept 

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = .3, random_state = 7)

#
# Fitting the training data and scoring the model
model.fit(X_train, y_train)
model.predict(X_test)
score = model.score(X_test,y_test)
drawPlane(model, X_test, y_test, "Accept(Room&Board,Enroll)", score)