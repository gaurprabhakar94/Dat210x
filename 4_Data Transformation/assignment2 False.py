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


scaleFeatures = False


# Loading up the dataset and removing any and all
# Rows that have a nan. 
df = pd.read_csv("kidney_disease.csv", sep=',', header= 0)
df = df.dropna(axis = 0)


# Create color coded labels because the actual label feature
# will be removed prior to executing PCA
labels = ['red' if i=='ckd' else 'green' for i in df.classification]


# Indexing to select columns
mdf = df.loc[:,['bgr','wc','rc']]


# Printing out and checking the dataframe's dtypes. 
print(mdf.dtypes)
mdf = mdf.apply(pd.to_numeric, args=('coerce',))
print(mdf.dtypes)
print(mdf)


# Checking the variance of every feature in your dataset.
print(mdf.var(axis=0))
print(mdf.describe())


if scaleFeatures: mdf = helper.scaleFeatures(mdf)



# Running PCA on the dataset and reducing it to 2 components

pca = PCA(n_components = 2)
print(pca.fit(mdf))
T = pca.transform(mdf) #data returns in a NumPy NDArray
print(T)



# Converting to a Pandas Dataframe.
#
# Note: Since we transformed via PCA, we no longer have column names.

ax = helper.drawVectors(T, pca.components_, mdf.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


