import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
import os
import numpy as np
os.chdir("Datasets")
matplotlib.style.use('ggplot') # Look Pretty

#Analysis based on the assumption that:
# On Weekends:
#   1. People probably don't go into work
#   2. They probably sleep in late on Saturday
#   3. They probably run a bunch of random errands, since they couldn't during the week
#   4. They should be home, at least during the very late hours, e.g. 1-4 AM
#
# On Weekdays:
#   1. People probably are at work during normal working hours
#   2. They probably are at home in the early morning and during the late night
#   3. They probably spend time commuting between work and home everyday

def showandtell(title=None):
  if title != None: plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
  plt.show()


def clusterInfo(model):
  print "Cluster Analysis Inertia: ", model.inertia_
  print '------------------------------------------'
  for i in range(len(model.cluster_centers_)):
    print "\n  Cluster ", i
    print "    Centroid ", model.cluster_centers_[i]
    print "    #Samples ", (model.labels_==i).sum() # NumPy Power

# Finding the cluster with the least # attached nodes
def clusterWithFewestSamples(model):
  # Ensuring there's at least on cluster...
  minSamples = len(model.labels_)
  minCluster = 0
  for i in range(len(model.cluster_centers_)):
    if minSamples > (model.labels_==i).sum():
      minCluster = i
      minSamples = (model.labels_==i).sum()
  print "\n  Cluster With Fewest Samples: ", minCluster
  return (model.labels_==minCluster)


def doKMeans(data, clusters=0):
  # Printing out the centroid locations and adding them onto the scatter plot. 
  df1 = pd.concat([data.TowerLon, data.TowerLat], axis = 1)
  kmeans = KMeans(n_clusters= clusters)
  kmeans.fit(df1)
  model = kmeans.predict(df1)
  centroids = kmeans.cluster_centers_
  ax.scatter(x = centroids[:, 0], y = centroids[:, 1], c = 'r', marker = 'x', s = 100)
  model = kmeans
  print(centroids)
  return model


#
# Loading up the dataset and take a peek at its head and dtypes.

df = pd.read_csv("CDR.csv", sep = ",", header = 0)
print(df.head())

# Converting the date and time attributes
df.CallDate = pd.to_datetime(df.CallDate, errors = 'coerce')
df.CallTime = pd.to_timedelta(df.CallTime, errors = 'coerce')


#
# Creating a unique list of of the phone-number values stored in the
# "In" column of the dataset
unique_numbers = df.In.unique()

#
# Creating a loop that filters and plots data for each user
#
for i in range(10):
    user1 = df[df.In == unique_numbers[i]]
    print ("\n\nExamining person: "+ str(i))

    #
    # Altering the slice so that it includes only Weekday (Mon-Fri) values and before 5pm
    user1 = user1.loc[~(user1['DOW'].isin(['Sun', 'Sat']))]
    user1 = user1[(user1['CallTime']<'17:00:00')]
    
    #
    # Plotting the Cell Towers the user is connected to
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(user1.TowerLon,user1.TowerLat, c='g', marker='o', alpha=0.2)
    ax.set_title('Weekday Calls before 5pm')
  

    # Running K-Means with K=3 or K=4. There really should only be a two areas of concentration. If there isn't
    # and multiple areas  are "hot", then we'll run with K=5, with the goal being that all centroids except two will
    # sweep up the annoying outliers and not-home, not-work travel occasions.
    model = doKMeans(user1, 5)
    print(unique_numbers[i])
    
    #
    # Printing out the mean CallTime value for the samples belonging to the cluster with the LEAST
    # samples attached to it. The cluster with the MOST samples will be work.
    # The cluster with the 2nd most samples will be home. And the K=3 cluster with the least samples
    # should be somewhere in between the two.
    midWayClusterIndices = clusterWithFewestSamples(model)
    midWaySamples = user1[midWayClusterIndices]
    print "    Its Waypoint Time: ", midWaySamples.CallTime.mean()
    
    
    ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=169, c='r', marker='x', alpha=0.8, linewidths=2)
    plt.show()