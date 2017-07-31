import pandas as pd
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import glob
from sklearn.manifold import Isomap
# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

import os
os.chdir("Datasets")


samples = [] 

#
# Write a for-loop appending each of the image to
# the list sample. 

directory_list1 = glob.glob('ALOI/32/*.png')
for filename in directory_list1: 							# iterating through file names 
    img = misc.imread(filename); 							# loading image to array with imread 
    samples.append( (img[::2, ::2] / 255.0).reshape(-1) ) 	# converting 1d and unitize



directory_list2 = glob.glob('ALOI/32i/*.png')

for filename in directory_list2: 							# iterating through file names 
    img = misc.imread(filename);							# loading image to array with imread 
    samples.append( (img[::2, ::2] / 255.0).reshape(-1) ) 	#converting 1d and unitize



marker_color = []

for i in range(len(directory_list1)):
    marker_color.append('b')
    
for _ in range(len(directory_list2)):
    marker_color.append('g')

#
# Converting the list to a dataframe
# 
print(len(samples))
df = pd.DataFrame(samples)
print(df)


#
# Implementing Isomap here. 
iso = Isomap(n_neighbors = 6, n_components =3)
iso.fit(df)
manifold = iso.transform(df)



#
# Creating a 2D Scatter plot to graph the manifold. 
plt.scatter(manifold[:, 0], manifold[:, 1], marker='o', c=marker_color)


#
# Creating a 3D Scatter plot to graph your manifold.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(manifold[:, 0], manifold[:, 1], manifold[:, 2], c=marker_color)


plt.show()