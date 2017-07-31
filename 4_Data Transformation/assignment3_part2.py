import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper
import os
from sklearn.decomposition import PCA

os.chdir("Datasets")

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


scaleFeatures = True


# Loading up the dataset and removing any and all
# Rows that have a nan.
df = pd.read_csv("kidney_disease.csv", sep=',', header= 0)
df = df.dropna(axis = 0)


# Create color coded labels because the actual label feature
# will be removed prior to executing PCA
labels = ['red' if i=='ckd' else 'green' for i in df.classification]


# Dropping the nominal features listed:
#       ['id', 'classification']
#
df = df.drop(labels = ['id','classification'], axis=1)
df = pd.get_dummies(df, columns = ['rbc', 'pc', 'pcc', 'ba','htn','dm','cad','appet', 'pe','ane'])
print(df)


# Printing out and checking the dataframe's dtypes. 
df = df.apply(pd.to_numeric, args=('coerce',))
print(df.dtypes)




# Checking the variance of every feature in your dataset.
print(df.var(axis=0))
print(df.describe())


if scaleFeatures: df = helper.scaleFeatures(df)



# Running PCA on the dataset and reducing it to 2 components
pca = PCA(n_components = 2)
print(pca.fit(df))
T = pca.transform(df) #data returns in a NumPy NDArray
print(T)

# Converting to a Pandas Dataframe.
#
# Note: Since we transformed via PCA, we no longer have column names.

ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


