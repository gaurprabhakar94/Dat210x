#
# This code is intentionally missing!
# Read the directions on the course lab page!
#

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
os.chdir("Datasets")
from sklearn.decomposition import PCA
from sklearn import manifold


#TODO: Load up the data set into a variable X,
#being sure to drop the name column.
#Splice out the status column into a variable y and delete it from X.
X = pd.read_csv("parkinsons.data", header=0)
y = X.status
X.drop(["name", "status"], inplace=True, axis = 1)
print(X)

#Perform a train/test split. 30% test group size, 
#with a random_state equal to 7.

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=7)
'''
#Create a SVC classifier. Don't specify any parameters, 
#just leave everything as default.
#Fit it against your training data and then score your testing data.

svc = SVC()
svc.fit(X_train, y_train)
score = svc.score(X_test, y_test)

print(score)
'''

#
# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation. Recall: when you do pre-processing,
# which portion of the dataset is your model trained upon? Also which portion(s)
# of your dataset actually get transformed?
#
# .. your code here ..

norm = Normalizer().fit(X_train)
maxabs = MaxAbsScaler().fit(X_train)
minmax = MinMaxScaler().fit(X_train)
stand = StandardScaler().fit(X_train)
robust = RobustScaler().fit(X_train)

#X_train = norm.transform(X_train) #79 #79
#X_test = norm.transform(X_test)

#X_train = maxabs.transform(X_train) #88 #88
#X_test = maxabs.transform(X_test)

#X_train = minmax.transform(X_train) #88 #88
#X_test = minmax.transform(X_test)

X_train = stand.transform(X_train) #0.94915254237288138
X_test = stand.transform(X_test) #0.93220338983050843

#X_train = robust.transform(X_train) #0.94915254237288138
#X_test = robust.transform(X_test) 0.9152542372881356



#
#TODO: Program a naive, best-parameter search by creating 
#nested for-loops. The outer for-loop should iterate a variable 
#C from 0.05 to 2, using 0.05 unit increments. 
#The inner for-loop should increment a variable gamma 
#from 0.001 to 0.1, using 0.001 unit increments. 
#As you know, Python ranges won't allow for float intervals, 
#so you'll have to do some research on NumPy ARanges,
#if you don't already know how to use them.
#
C = 0.05
gamma = 0.001
best_score = 0
c_range = np.arange(0.05,2.05, 0.05)
gamma_range = np.arange(0.001, .101, 0.001)
pca_range = np.arange(4,15,1)
iso_range_neighbors = np.arange(2,6,1)
iso_range_components = np.arange(4,7,1)
n_components = 0
n_neighbors = 0



'''
for a in pca_range:
    pca = PCA(n_components = a)
    pca.fit(X_train)
    X_Ltrain = pca.transform(X_train)
    X_Ltest = pca.transform(X_test)
'''

for a in iso_range_neighbors:
    for b in iso_range_components:
        iso = manifold.Isomap(n_neighbors = a, n_components = b)
        iso.fit(X_train)
        X_Ltrain = iso.transform(X_train)
        X_Ltest = iso.transform(X_test)

        for i in c_range:
            for j in gamma_range:
                    #
                    #create an SVC model and pass in the C and gamma parameters its 
                    #class constructor. Train and score the model appropriately. 
                    #If the current best_score is less than the model's score, 
                    #update the best_score being sure to print it out, along with 
                    #the C and gamma values that resulted in it.
                    #
                model = SVC(C=i, gamma=j)
                model.fit(X_Ltrain, y_train)
                score = model.score(X_Ltest, y_test)
                if(best_score<score):
                    best_score = score
                    C = i
                    gamma = j
                    n_components = b
                    n_neighbors = a

print("best score:", best_score)
print("best C:", C)
print("best gamma:", gamma)
print("best n_components:", n_components)
print("best n_neighbors:", n_neighbors)


