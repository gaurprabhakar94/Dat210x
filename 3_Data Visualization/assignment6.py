import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir("Datasets")


#
# Loading up the Seeds Dataset into a Dataframe
df = pd.read_csv("wheat.data", sep =',', header = 0)

#
# Dropping the 'id' feature
df.drop('id', axis = 1, inplace = True)

#
# Computing the correlation matrix of your dataframe
print(df.corr())

#
# Graphing the correlation matrix using imshow or matshow
# 
plt.imshow(df.corr(), cmap = plt.cm.Blues, interpolation = 'nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]

plt.xticks(tick_marks, df.columns, rotation = 'vertical')
plt.yticks(tick_marks, df.columns)

plt.show()