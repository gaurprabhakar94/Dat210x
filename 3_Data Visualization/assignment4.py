import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from pandas.tools.plotting import parallel_coordinates
os.chdir("Datasets")
# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# Loading up the Seeds Dataset into a Dataframe
df = pd.read_csv("wheat.data", sep=',', header = 0)
print(df)

#
# Drop the 'id','area' and 'perimeter' features
df.drop(df.columns[[0,1,2]], axis = 1, inplace = True)
print(df)

#
# Ploting a parallel coordinates chart grouped by
# the 'wheat_type' feature.
parallel_coordinates(df,'wheat_type', alpha =.4)
plt.show()