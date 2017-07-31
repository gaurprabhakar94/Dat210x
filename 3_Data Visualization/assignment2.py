import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
os.chdir("Datasets")

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# Loading up the Seeds Dataset into a Dataframe
df = pd.read_csv("wheat.data", sep=',', header=0)


#
# Creating a 2d scatter plot that graphs the
# area and perimeter features
df.plot.scatter(x='area',y='perimeter')

#
# Creating a 2d scatter plot that graphs the
# groove and asymmetry features
df.plot.scatter(x='groove',y='asymmetry')

#
# Creating a 2d scatter plot that graphs the
# compactness and width features
# 
df.plot.scatter(x='compactness',y='width',marker='^')

plt.show()