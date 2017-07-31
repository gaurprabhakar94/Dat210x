import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import os
# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

os.chdir("Datasets")
#
# Loading up the Seeds Dataset into a Dataframe
df = pd.read_csv("wheat.data", sep=',', header = 0)
print(df)

fig = plt.figure()

#
# Creating a new 3D subplot using fig. 
# Then use the subplot to graph a 3D scatter 
# plot using the area, perimeter and asymmetry features. 
#
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('Area')
ax.set_ylabel('Perimeter')
ax.set_zlabel('Asymmetry')
ax.scatter(df.area, df.perimeter, df.asymmetry, c='red')


fig = plt.figure()
#
# Creating a new 3D subplot using fig.
# Then use the subplot to graph a 3D scatter plot using the width,
# groove and length features. 
# 
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('Width')
ax.set_ylabel('Groove')
ax.set_zlabel('Length')
ax.scatter(df.width, df.groove, df.length, c='green')

plt.show()