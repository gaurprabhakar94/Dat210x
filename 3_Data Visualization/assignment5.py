import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from pandas.tools.plotting import andrews_curves

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

os.chdir("Datasets")

#
# Loading up the Seeds Dataset into a Dataframe
df = pd.read_csv("wheat.data", sep=',', header = 0)
print(df)


#
# Dropping the 'id', 'area' and 'perimeter' features
df.drop('id', axis = 1, inplace = True)
print(df)

#
# Plotting a parallel coordinates chart grouped by
# the 'wheat_type' feature.
andrews_curves(df,'wheat_type', alpha =0.4)

plt.show()