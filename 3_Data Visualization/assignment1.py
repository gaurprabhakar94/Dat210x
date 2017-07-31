import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


os.chdir("Datasets")

#
# Loading up the Seeds Dataset into a Dataframe
#
df = pd.read_csv("wheat.data", sep=',', header=0)
print(df)

#
# Creating a slice of the dataframe 
#
s1 = df.ix[:,['area','perimeter']]
print(s1)

#
# Creating another slice of the dataframe
s2 = df[['groove','asymmetry']]
print(s2)


#
# Creating a histogram plot using the first slice,
# and another histogram plot using the second slice.
# 
s1.plot.hist(alpha = 0.75)
s2.plot.hist(alpha = 0.75)

# Display the graphs:
plt.show()

