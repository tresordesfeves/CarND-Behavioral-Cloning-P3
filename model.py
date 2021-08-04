#model.05-Loss0.0104-valLoss0.0094.h5        
import os
import csv
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Cropping2D

from keras import initializers 

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from skimage.transform import rescale, resize, downscale_local_mean

from sklearn.utils import shuffle
from keras import initializers 
from scipy import ndimage
import cv2 
import pickle
import numpy as np
from math import ceil
import matplotlib; matplotlib.use('agg')
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


import glob
import sklearn
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint,EarlyStopping

import pandas as pd

from sklearn.utils import resample
from numpy import zeros

from sklearn.utils import resample


df_provided=pd.read_csv("/home/workspace/CarND-Behavioral-Cloning-P3/driving_log_provided.csv")
df_provided=shuffle(df_provided)
print(df_provided.head())


df_all=df_provided

# spliting the data set between training and validation 
df_train, df_validation_samples = train_test_split(df_all, test_size=0.1)
validation_samples=df_validation_samples.values.tolist()
train_samples=df_train.values.tolist()

# Callbacks to save all weigts for all epoch 
# and later pick the set of weights with the lowest validation value 
# weights will be save in "bestModelFolder"
checkpoint = ModelCheckpoint(filepath='bestModelFolder/model.{epoch:02d}-Loss{loss:.4f}-valLoss{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)


# a function to draw a line chart to visualize training and validation values for each epoch
def visualize_loss_history(history) :
### plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
    plt.savefig('lossHistory.png')

t
# a custom generator that augment the center image data set 
# with left, right, flipped center, flipped left , flipped right  images and steering
# The generator only load a batch size number of images in memory ( multiply bt 6 after augmentation )

def generator(list_of_csv_lines, batch_size=32, training=True):

    #all_blacks=zeros(64, 64, dtype=int)
    #all_blacks = np.zeros((160, 160), dtype=int)

    while 1: # Loop forever so the generator never terminates
        
        if training:
            

            print ("=========================== EPOCH IN WHILE LOOP=")

            df = pd.DataFrame(list_of_csv_lines,columns=columns)    
           
            df_train=shuffle(df)

            train_samples =df_train.values.tolist()
####
        else :

            train_samples=list_of_csv_lines

        num_samples = len(train_samples) 

        train_samples=shuffle(train_samples) # samples are shuffled at each Epoch  
        
        for offset in range(0, num_samples, batch_size): # looping through the set
            
            batch_samples = train_samples[offset:offset+batch_size]

            images = []
            angles = []
            
            X_center_BGR=[ cv2.imread((bs[0].strip()) ) for bs in batch_samples]

            X_left_BGR=[ cv2.imread((bs[1].strip()) ) for bs in batch_samples]

            X_right_BGR=[ cv2.imread((bs[2].strip()) ) for bs in batch_samples]

            
            #converstion BGR to YUV
            X_center=[ cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV) for bgr in X_center_BGR]
            X_left=[ cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV) for bgr in X_left_BGR]
            X_right=[ cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV) for bgr in X_right_BGR]
            
            correction = 0.2 # this is a parameter to tune steering for side cameras pictures
         
            y_center = [ float(bs[3]) for bs in batch_samples]
            y_left = [ float(bs[3])+correction for bs in batch_samples]
            y_right = [ float(bs[3])-correction for bs in batch_samples]


            # augment the se with flipped images 
            X_flip_center=[cv2.flip(x,1) for x in X_center ]
            X_flip_left=[cv2.flip(x,1) for x in X_left ]
            X_flip_right=[cv2.flip(x,1) for x in X_right ]
            y_flip_center=[-y for y in y_center ]
            y_flip_left=[-y for y in y_left ]
            y_flip_right=[-y for y in y_right ]

            images=X_center + X_left + X_right + X_flip_center + X_flip_left + X_flip_right

            angles= y_center + y_left + y_right + y_flip_center + y_flip_left + y_flip_right

            X_train = np.array(images)
            y_train = np.array(angles)

            X_train = X_train.reshape(-1,160, 320, 3)
            
            #each batch is actually 6 times the size of "batch_size"
            yield sklearn.utils.shuffle(X_train, y_train)



# compile and train the model using the generator function


batch_size=256

train_generator = generator(train_samples, batch_size=batch_size, training=True)

validation_generator = generator(validation_samples, batch_size=batch_size, training=False)


#############################################################################################################
# model 
dOut_rate=0.2
Aventador= Sequential()
Aventador.add(Lambda(lambda x: x/255 -0.5, input_shape=(160,320,3)))

Aventador.add(Cropping2D(cropping=((70,25),(0,0))))

Aventador.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))

Aventador.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))

Aventador.add(Conv2D(48, (5,5), activation="relu", strides=(2, 2)))

Aventador.add(Conv2D(64, (3, 3), activation="relu"))

Aventador.add(Conv2D(64, (3, 3), activation="relu"))
Aventador.add(Flatten())

Aventador.add(Dense(100))

Aventador.add(Dropout(dOut_rate))

Aventador.add(Dense(50))

Aventador.add(Dropout(dOut_rate))

Aventador.add(Dense(10))
Aventador.add(Dropout(dOut_rate))
Aventador.add(Dense(1))
Aventador.compile(loss="mse", optimizer="adam",metrics = ["accuracy"])

Aventador.load_weights("bestModelFolder/model.06-Loss0.0205-valLoss0.0180.h5") 


# training and saving  the weights for each epoch (callbacks=[checkpoint])
history_object=Aventador.fit_generator(train_generator,
    steps_per_epoch=ceil(len_train_set/batch_size),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples)/batch_size),
    epochs=7, verbose=1, callbacks=[checkpoint]) 

#Aventador.save('geneRC')

#creating the line chart of the loss history 
visualize_loss_history(history_object)
