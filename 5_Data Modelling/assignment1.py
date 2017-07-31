import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.cluster import KMeans
import pandas as pd

# Look Pretty
matplotlib.style.use('ggplot')
plt.style.use('ggplot')

os.chdir("Datasets")

def doKMeans(df):
  #
  # Plotting data with a '.' marker, with 0.3 alpha at the Longitude,
  # and Latitude locations in the dataset. Longitude = x, Latitude = y
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)

  #
  # Filtering df to look at Longitude and Latitude,
  df = df.filter(items=['Longitude', 'Latitude'])
  
  #
  # Using K-Means to try and find cluster centers
  model = KMeans(n_clusters = 7,  init = 'random', n_init = 60, max_iter = 360, random_state = 43)
  model.fit_predict(df)
  
  #
  # Printing and plotting the centroids
  centroids = model.cluster_centers_
  ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
  print centroids



#
# Loading the dataset
#
df = pd.read_csv("Crimes_-_2001_to_present.csv", header = 0, sep=',')

#
# Dropping any ROWs with nans in them
df.dropna(axis=0, inplace = True)


#
# Printing out the dtypes and checking for errors
print(df.dtypes)

#
# Changing the dtype of date from string to date
df.Date = pd.to_datetime(df.Date, errors = 'coerce')
print(df.dtypes)

# Printing & Plotting the data
doKMeans(df)


#
# Filtering out the data
df2 = df[df.Date > '2011-01-01']



# Printing & Plotting the data
doKMeans(df2)
plt.title("Dates after 2011")
plt.show()


