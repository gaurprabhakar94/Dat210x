import pandas as pd
import os
os.chdir('Datasets')

# Loding up the dataset
# 
df = pd.read_csv("servo.data", sep=',', names = ['motor', 'screw', 'pgain', 'vgain', 'class'])
print(df)
print("\n")

# Creating a slice that contains all entries
# having a vgain equal to 5. Then printing the 
# length of (# of samples in) that slice:
#
print(len(df[df.vgain==5]))
print("\n")


# Creating a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then printing the length of (# of
# samples in) that slice:
#
print(len(df[(df.motor=='E') & (df.screw=='E')]))
print("\n")

# Creating a slice that contains all entries
# having a pgain equal to 4. Then finding the mean vgain
# value for the samples in that slice. 
#
print(df.vgain[df.pgain==4].mean())
print("\n")


print(df.dtypes)