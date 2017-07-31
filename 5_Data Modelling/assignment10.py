import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import os
from sklearn import linear_model

#
# This is how much of the audio file will
# be provided, in percent. The remaining percent of the file will
# be generated via linear extrapolation.
Provided_Portion = 0.25



# Regular python empty list
zero = []


#
# Looping through the dataset and loading up all 50 of the 0_jackson*.wav
# files 
# .read() returns a tuple 

for filename in os.listdir('Datasets/free-spoken-digit-dataset-master/recordings'):
    if filename.startswith('0_jackson'):
        sample = os.path.join('Datasets/free-spoken-digit-dataset-master/recordings', filename)
        sample_rate, audio_data = wavfile.read(sample)
        zero.append(audio_data)
print(zero)



# Hard chopping all audio clips to be the same length.
# Dropping all Nans on the Y axis here and converting the dataset into an
# NDArray 
zero = pd.DataFrame(data = zero, dtype = np.int16)
zero.dropna(axis=1, inplace = True)
zero = zero.values



# 'zero' is currently shaped [n_samples, n_audio_samples],
n_audio_samples = zero.shape[1]
print(n_audio_samples)


#
# Creating your linear regression model here
model = linear_model.LinearRegression()


from sklearn.utils.validation import check_random_state
rng   = check_random_state(7)  
random_idx = rng.randint(zero.shape[0])
test  = zero[random_idx]
train = np.delete(zero, [random_idx], axis=0)


# 
# Printing out the shape of train, and the shape of test

print(train.shape, test.shape)


#
# Saving the original 'test' clip part 
wavfile.write('Original Test Clip.wav', sample_rate, test)




# Grabbing the FIRST Provided_Portion * n_audio_samples audio features 
# from test

X_test = test[:int(Provided_Portion*n_audio_samples)]


# Grabbing the *remaining* audio features and storing it in y_test.

y_test = test[int(Provided_Portion*n_audio_samples):]



# 
# Duplicating the same above process for X_train, y_train.

X_train = train[:,:int(Provided_Portion*n_audio_samples)]
y_train = train[:,int(Provided_Portion*n_audio_samples):]


# .reshape(1, -1) turns [n_features] into [1, n_features].
# .reshape(-1, 1) turns [n_samples] into [n_samples, 1].

X_test = X_test.reshape(1,-1)
y_test = y_test.reshape(1,-1)

#
# Fitting model using training data and label
model.fit(X_train, y_train)

# 
# Using the model to predict the 'label' of X_test. 
y_test_prediction = model.predict(X_test)


y_test_prediction = y_test_prediction.astype(dtype=np.int16)


# Checking the accuracy score
score = model.score(X_test, y_test)
print "Extrapolation R^2 Score: ", score


#
# Taking the first Provided_Portion portion of the test clip and stitching that
# together with the abomination the predictor model generated
# and then saving the completed audio clip
completed_clip = np.hstack((X_test, y_test_prediction))
wavfile.write('Extrapolated Clip.wav', sample_rate, completed_clip[0])
