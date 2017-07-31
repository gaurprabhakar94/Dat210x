import pandas as pd
import os
os.chdir('Datasets')

# Loading up the dataset
#

df = pd.read_csv("tutorial.csv", sep=',')
print(df)
print("\n")


# Printing the results of the .describe() method
#
print(df.describe())
print("\n")


# Indexing the dataframe with: [2:4,'col3']
# and then printing the value
print(df.loc[2:4,'col3'])
