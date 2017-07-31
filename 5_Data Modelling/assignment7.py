import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# True if test with PCA
Test_PCA = False


def plotDecisionBoundary(model, X, y):
  print "Plotting..."
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
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
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plotting the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plotting the testing points
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()


# 
# Loading the dataset, identifying nans, and setting proper headers.

df = pd.read_csv("Datasets/breast-cancer-wisconsin.data", header = None, names = ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status'])
print(df.head())
print(df.dtypes)

df.nuclei = pd.to_numeric(df.nuclei, errors="coerce")

print(df.dtypes)
print(df.isnull().values.any())


# 
# Copying out the status column into a slice
status = df.status
print(len(df.status))
print(len(status))
df.drop(["status", "sample"], inplace = True, axis = 1)


#
# Extract labels and replace nan values
df.nuclei = df.nuclei.fillna(df.nuclei.mean())
print(df.isnull().values.any())



data_train, data_test, label_train, label_test = train_test_split(df, status, test_size = 0.5, random_state =7)



# Uncomment the only one set at a time and leave them commented to run without preprocessing

norm = Normalizer().fit(data_train)
maxabs = MaxAbsScaler().fit(data_train)
minmax = MinMaxScaler().fit(data_train)
stand = StandardScaler().fit(data_train)
robust = RobustScaler().fit(data_train)

#data_train = norm.transform(data_train)
#data_test = norm.transform(data_test)

#data_train = maxabs.transform(data_train)
#data_test = maxabs.transform(data_test)

#data_train = minmax.transform(data_train)
#data_test = minmax.transform(data_test)

#data_train = stand.transform(data_train)
#data_test = stand.transform(data_test)

#data_train = robust.transform(data_train)
#data_test = robust.transform(data_test)



model = None

if Test_PCA:
  print "Computing 2D Principle Components"
  model = PCA(n_components=2)

else:
  print "Computing 2D Isomap Manifold"
  model = manifold.Isomap(n_components = 2, n_neighbors = 5)


# Training the model
model.fit(data_train)
data_train = model.transform(data_train)
data_test = model.transform(data_test)

# 
# Implementing and training KNeighborsClassifier on the projected 2D
# training data here.
knmodel = KNeighborsClassifier(n_neighbors = 5, weights='uniform')
knmodel.fit(data_train, label_train)
predict = knmodel.predict(data_test)

print(knmodel.predict_proba(data_test))
print(accuracy_score(label_test, predict))
plotDecisionBoundary(knmodel, data_test, label_test)