import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import os
os.chdir("Datasets")


matplotlib.style.use('ggplot') # Look Pretty


def plotDecisionBoundary(model, X, y):
  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.6
  resolution = 0.0025
  colors = ['royalblue','forestgreen','ghostwhite']

  # Calculating the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Creating a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plotting the contour map
  cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

  # Plotting the test original points
  for label in range(len(np.unique(y))):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

  p = model.get_params()
  plt.axis('tight')
  plt.title('K = ' + str(p['n_neighbors']))


# 
# Loading up the dataset 
# 
X =pd.read_csv("wheat.data", sep=',', header=0)
print(X.head())


y = X.wheat_type.copy()
X.drop(['wheat_type','id'], inplace = True, axis = 1)


# Performing an ordinal conversion
y = y.astype("category").cat.codes


#
# Filling nans with the mean of the feature
#
X.compactness.fillna(X.compactness.mean(), inplace = True)
X.width.fillna(X.width.mean(), inplace = True)
X.groove.fillna(X.groove.mean(), inplace = True)

#
# plitting X into training and testing data sets
data_train,  data_test, label_train, label_test = train_test_split(X, y, test_size = 0.33, random_state = 1)


# Fit training data against Normalizer()
nor = Normalizer()
nor.fit(data_train)


#
# Transforming both the training AND
# testing data.
#
X_trainnor = nor.transform(data_train)
X_testnor = nor.transform(data_test)


#
# Creating a PCA transformation and fitting it against the training data, and then
# transforming  the training and testing features 
pca = PCA(n_components = 2)
pca.fit(X_trainnor)

T_trainpca = pca.transform(X_trainnor)
T_testpca = pca.transform(X_testnor)


#
# Creating and training a KNeighborsClassifier.
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(T_trainpca, label_train)

plotDecisionBoundary(knn, T_trainpca, label_train)


# Displaying the accuracy score of the test data/labels
accuracy = knn.score(T_testpca, label_test)
print(accuracy)

plt.show()

