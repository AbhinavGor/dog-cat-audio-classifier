dataset='/content/drive/MyDrive/full_dataset'

import struct

class WavFileHelper():
    
    def read_file_properties(self, filename):

        wave_file = open(filename,"rb")
        
        riff = wave_file.read(12)
        fmt = wave_file.read(36)
        
        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I",sample_rate_string)[0]
        
        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H",bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)

import os
import pandas as pd
import librosa
import librosa.display
wavfilehelper = WavFileHelper()
audiodata=[]

for i in os.listdir(dataset):
  data = wavfilehelper.read_file_properties(dataset+"/"+i)
  audiodata.append(data)
audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])

print(audiodf.num_channels.value_counts(normalize=True))

print(audiodf.sample_rate.value_counts(normalize=True))

print(audiodf.bit_depth.value_counts(normalize=True))

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
     
    return mfccsscaled

print(audiodf.head())

features=[]
import numpy as np
for i in os.listdir(dataset):
  class_label=i[:3]
  data=extract_features(dataset+'/'+i)
  features.append([data,class_label])

df_features=pd.DataFrame(features,columns=['feature','class_label'])

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
# Convert features and corresponding classification labels into numpy arrays
X = np.array(df_features.feature.tolist())
y = np.array(df_features.class_label.tolist())

# Encode the classification labels 
le = LabelEncoder()
y2 = to_categorical(le.fit_transform(y)) 

# split the dataset 
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state = 42)

import matplotlib.pyplot as plt
X_train.shape

X_test.shape

n_rows=40
n_cols=1
n_channels=1
X_train=X_train.reshape(X_train.shape[0],n_rows,n_cols,n_channels)
X_test=X_test.reshape(X_test.shape[0],n_rows,n_cols,n_channels)

X_test.shape

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=1, input_shape=(n_rows, n_cols, n_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Conv2D(filters=32, kernel_size=1, activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Flatten())

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(X_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 50
num_batch_size = 8
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

features=[]
import numpy as np
data=extract_features('/content/drive/MyDrive/full_dataset/dog_barking_99.wav')
features.append([data])
df_features=pd.DataFrame(features,columns=['feature'])
# Convert features and corresponding classification labels into numpy arrays
X = np.array(df_features.feature.tolist())
X=X.reshape(X.shape[0],n_rows,n_cols,n_channels)
model.predict(X)
