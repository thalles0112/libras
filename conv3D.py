from numpy.lib.arraypad import pad
from utils import Utils
import numpy as np
import cv2 as cv
from time import sleep

TRAIN = True



traindata = Utils.loadVideo('/home/thalles/Documents/PROJETO INTEGRADOR/DATABASE', ['oi', 'boa', 'tarde'])
x, y = Utils.processVideoMOD(traindata, 20, (200, 200))


x = np.array(x, 'float16')
x = np.expand_dims(x, 0)

y = np.array(y, 'float16')
y = np.expand_dims(y, 0)



from tensorflow.keras import layers
from tensorflow.keras import Sequential
import tensorflow as tf

if TRAIN:
        
    model3D = Sequential([
        layers.Input((596, 200, 200,1), 3, dtype='float16') ,
        layers.Conv3D(2, (3,3,3), padding='valid', data_format='channels_last'),
        layers.Flatten(),
        layers.Dense(32, 'relu'),
        layers.Dense(3, 'softmax')

    ])

    model3D.compile('rmsprop', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), ['accuracy'],)


    model3D.fit(x, y, 3, epochs=4)

    #model3D.save('model3D caralho')
