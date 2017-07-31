import pandas as pd
import numpy as np
import scipy.io
import random, math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os
os.chdir("Datasets")

from sklearn.decomposition import PCA

from sklearn import manifold

def Plot2D(T, title, x, y, num_to_plot=40):
  # This method picks a bunch of random images to plot onto the chart
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
  for i in range(num_to_plot):
    img_num = int(random.random() * num_images)
    x0, y0 = T[img_num,x]-x_size/2., T[img_num,y]-y_size/2.
    x1, y1 = T[img_num,x]+x_size/2., T[img_num,y]+y_size/2.
    img = df.iloc[img_num,:].reshape(num_pixels, num_pixels)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

  # Plotting the full 2D scatter plot
  ax.scatter(T[:,x],T[:,y], marker='.',alpha=0.7)



def Plot3D(T, title, x, y, z, num_to_plot=40):
    # This method picks a bunch of random images to plot onto the chart:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('Component: {0}'.format(x))
    ax.set_ylabel('Component: {0}'.format(y))
    ax.set_zlabel('Component: {0}'.format(z))
    x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
    y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
    z_size = (max(T[:,z]) - min(T[:,z])) * 0.08
    for i in range(num_to_plot):
        img_num = int(random.random() * num_images)
        x0, y0, z0 = T[img_num,x]+x_size/2., T[img_num,y]+y_size/2., T[img_num,z]+z_size/2.
        x1, y1, z1 = T[img_num,x]-x_size/2., T[img_num,y]-y_size/2., T[img_num,z]-z_size/2.
        
        
        img = df.iloc[img_num,:].reshape(num_pixels, num_pixels)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    # Plotting the 3D full scatter plot
    ax.scatter(T[:,x],T[:,y], T[:,z], marker='.',alpha=0.7)


# importing .mat files
mat = scipy.io.loadmat('face_data.mat')
df = pd.DataFrame(mat['images']).T
num_images, num_pixels = df.shape
num_pixels = int(math.sqrt(num_pixels))

# Rotating the pictures
for i in range(num_images):
  df.loc[i,:] = df.loc[i,:].reshape(num_pixels, num_pixels).T.reshape(-1)


#
# Implementing PCA here
#
pca = PCA(n_components = 3)
pca.fit(df)
T = pca.transform(df)

Plot2D(T, 'PCA 0 and 1 components', 0, 1)
Plot2D(T, 'PCA 1 and 2 components', 1, 2)


#
# Implementing Isomap here. 

iso = manifold.Isomap(n_components = 3, n_neighbors = 3)
iso.fit(df)
manifold = iso.transform(df)

Plot2D(manifold, 'Isomap 0 and 1 components', 0, 1)
Plot2D(manifold, 'Isomap 1 and 2 components', 1, 2)

#
# Plotting in 3D

Plot3D(T, title = "PCA with 3 components", x =0, y = 1, z=2)
Plot3D(manifold, title = "Isomap with 3 components", x=0, y=1, z=2)

plt.show()
