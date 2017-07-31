import pandas as pd
import numpy as np
import os

os.chdir("Datasets")

#
# Loading up the dataset, setting correct header labels.
#
df = pd.read_csv("census.data", sep = ',', header = None, names = ['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification'], na_values = '?')
print(df)


#
# Looking through the dataset to get a
# feel for it before proceeding!
#
print(df['capital-gain'].unique())
print(df.dtypes)

#
# Looking through the data and identifying any potential categorical
# features. 
education_ordered = ['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Bachelors','Masters','Doctorate']
df.education = df.education.astype("category",ordered = True, categories = education_ordered).cat.codes
df = pd.get_dummies(df , columns = ['race','sex','classification'])

print(df)