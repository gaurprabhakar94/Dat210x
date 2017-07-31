****************************************************************

Assignment 1

Experimented with a real life armadillo sculpture scanned using a Cyberware 3030 MS 3D scanner at Stanford University. The sculpture is available as 
part of their 3D Scanning Repository, and is a very dense 3D mesh consisting of 172974 vertices! Used 3D Binary mesh loaded from Plyfile and converted 
into a Pandas dataframe for your ease of manipulation. Reduced the dimensionality from three to two using PCA and RandomizedPCA to cast a shadow of the data onto its 
two most important principal components. Then rendered the resulting 2D scatter plot.

****************************************************************

Assignment 2

Experimented with a subset of UCI's Chronic Kidney Disease data set, a collection of samples taken from patients in India over a two month period, 
some of whom were in the early stages of the disease. Cleaned the data, filtered it and gave the option of performing feature scaling or not. The feature 
scaling code is in file Assignment2_helper.py.
Then depending upon the choice performed PCA (Assignment2 True) or didn't perform (Assignment2 False). Then visualized the results on a scatter plot.

****************************************************************

Assignment 3

Used the same dataset as the previous assignment. Cleaned the data, and instead of filtering the data, dropped all nominal features and gave the option of 
performing feature scaling or not. The feature scaling code is in file Assignment2_helper.py. Then depending upon the choice performed PCA (Assignment3_part1)
Then visualized the results on a scatter plot.

For the second part, dropped only 2 nominal features and exploded the rest. Then depending upon the choice performed PCA (Assignment3_part2)
Then visualized the results on a scatter plot.

****************************************************************

Assignment 4

Used the Joshua Tenenbaum's original dataset for nonlinear dimensionality reduction, from December 2000. It consists of 698 samples of 4096-dimensional vectors. 
These vectors are the coded brightness values of 64x64-pixel heads that have been rendered facing various directions and lighted from many angles. 
Applyed both PCA and Isomap to theese 698 raw images to derive 2D principal components and a 2D embedding of the data's intrinsic geometric structure.
Projected both onto a 2D and 3D scatter plot, with a few superimposed face images on the associated samples.

****************************************************************

Assignment 5

Used the ALOI, Amsterdam Library of Object Images, dataset that hosts a huge collection of 1000 small objects that were photographed in such a controlled
environment, by systematically varying the viewing angle, illumination angle, and illumination color for each object separately. Ran and tested isomap on 
this carefully constructed dataset, visually confirmed its effectiveness, and gained a deeper understanding of how and why each parameter acts the way it does.

****************************************************************