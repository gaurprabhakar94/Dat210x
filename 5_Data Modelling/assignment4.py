import numpy as np
import pandas as pd
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib



PLOT_TYPE_TEXT = False    # For indices
PLOT_VECTORS = True       # For original features in P.C.-Space


matplotlib.style.use('ggplot') # Look Pretty
c = ['red', 'green', 'blue', 'orange', 'yellow', 'brown']

def drawVectors(transformed_features, components_, columns, plt):
  num_columns = len(columns)

  # This function will project the original feature onto the principal component feature-space, 
  # Scaling the principal components by the max value in
  # the transformed set belonging to that component
  xvector = components_[0] * max(transformed_features[:,0])
  yvector = components_[1] * max(transformed_features[:,1])

  # Sorting each column by its length. 
  import math
  important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
  important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
  print "Projected Features by importance:\n", important_features

  ax = plt.axes()
  for i in range(num_columns):
    # Using an arrow to project each original feature as a
    # labeled vector on the principal component axes
    plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75, zorder=600000)
    plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75, zorder=600000)
  return ax
    

def doPCA(data, dimensions=2):
  from sklearn.decomposition import PCA
  import sklearn
  print sklearn.__version__
  model = PCA(n_components=dimensions, svd_solver='randomized', random_state=7)
  model.fit(data)
  return model


def doKMeans(data, clusters=0):
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters = clusters)
  kmeans.fit(data)
  model = kmeans.predict(data)
  model = kmeans
  return model.cluster_centers_, model.labels_



import os
os.chdir("Datasets")

df = pd.read_csv("Wholesale customers data.csv", sep=',', header = 0)
# Setting Nans to 0
df.fillna(0)


df.drop(['Channel','Region'], axis = 1, inplace = True)


df.plot.hist()

# Removing top 5 and bottom 5 samples for each column to reduce big gaps
drop = {}
for col in df.columns:
  # Bottom 5
  sort = df.sort_values(by=col, ascending=True)
  if len(sort) > 5: sort=sort[:5]
  for index in sort.index: drop[index] = True # Just store the index once

  # Top 5
  sort = df.sort_values(by=col, ascending=False)
  if len(sort) > 5: sort=sort[:5]
  for index in sort.index: drop[index] = True # Just store the index once

#
# Dropping rows by index. 
print "Dropping {0} Outliers...".format(len(drop))
df.drop(inplace=True, labels=drop.keys(), axis=0)



#
# Un-commenting one line at a time before running the code
T = preprocessing.StandardScaler().fit_transform(df)
#T = preprocessing.MinMaxScaler().fit_transform(df)
#T = preprocessing.MaxAbsScaler().fit_transform(df)
#T = preprocessing.Normalizer().fit_transform(df)

T = df # No Change


# KMeans
n_clusters = 3
centroids, labels = doKMeans(T, n_clusters)


#
# Printing out the centroids.
print(centroids)

# Projecting the centroids and samples into the new 2D feature space
display_pca = doPCA(T)
T = display_pca.transform(T)
CC = display_pca.transform(centroids)


# Visualizing all the samples and giving them the color of their cluster label
fig = plt.figure()
ax = fig.add_subplot(111)
if PLOT_TYPE_TEXT:
  # Plotting the index of the sample
  for i in range(len(T)): ax.text(T[i,0], T[i,1], df.index[i], color=c[labels[i]], alpha=0.75, zorder=600000)
  ax.set_xlim(min(T[:,0])*1.2, max(T[:,0])*1.2)
  ax.set_ylim(min(T[:,1])*1.2, max(T[:,1])*1.2)
else:
  # Plotting a regular scatter plot
  sample_colors = [ c[labels[i]] for i in range(len(T)) ]
  ax.scatter(T[:, 0], T[:, 1], c=sample_colors, marker='o', alpha=0.2)


# Plotting the Centroids as X's
ax.scatter(CC[:, 0], CC[:, 1], marker='x', s=169, linewidths=3, zorder=1000, c=c)
for i in range(len(centroids)): ax.text(CC[i, 0], CC[i, 1], str(i), zorder=500010, fontsize=18, color=c[i])


# Displaying the feature vectors
if PLOT_VECTORS: drawVectors(T, display_pca.components_, df.columns, plt)


# Adding the cluster label back into the dataframe
df['label'] = pd.Series(labels, index=df.index)
print df

plt.show()
