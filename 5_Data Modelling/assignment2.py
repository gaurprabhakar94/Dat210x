import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
matplotlib.style.use('ggplot') # Look Pretty
import os

os.chdir("Datasets")


#Analysis based on the following assumption
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
  


df = pd.read_csv("CDR.csv", sep = ",", header = 0)
print(df.head())

df.CallDate = pd.to_datetime(df.CallDate, errors = 'coerce')
df.CallTime = pd.to_timedelta(df.CallTime, errors = 'coerce')

#
# Getting a distinct list of "In" phone numbers
#
inphono = df.In.unique()

# 
# Creating a loop that filters and plots data for each user
#
for i in range(10):
    user1 = df[df.In == inphono[i]]

# PLotting all the call locations
    user1.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title=(str(('Call Locations for user '+str(i)))))


# Examining records that came in on weekends (sat/sun).
    user1 = user1.loc[user1['DOW'].isin(['Sat', 'Sun'])]


#
# Filtering it down for calls that came in either before 6AM OR after 10pm (22:00:00).
    user1 = user1.loc[(user1['CallTime']<'06:00:00')|(user1['CallTime']>'22:00:00')]
    print(len(user1))


# In the plot it's likely that wherever records are bunched up it is probably near where the
# caller's residence is
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(user1.TowerLon,user1.TowerLat, c='g', marker='o', alpha=0.2)
    ax.set_title(str('Weekend Calls between 10pm to 6am for user '+ str(i)))



#
# Running K-Means with a K=1. There really should only be a single area of concentration. If there isn't
# and multiple areas are "hot", then we'll run with K=2, with the goal being that one of the centroids will
# sweep up the annoying outliers
    user1 = pd.concat([user1.TowerLon, user1.TowerLat], axis = 1)
    kmeans = KMeans(n_clusters = 1)
    kmeans.fit(user1)
    labels = kmeans.predict(user1)
    centroids = kmeans.cluster_centers_
    print(centroids)
    print(inphono[i])
    
	#Printing out the centroid locations and add them onto the scatter plot.
    ax.scatter(x = centroids[:, 0], y = centroids[:, 1], c = 'r', marker = 'x', s = 100)


