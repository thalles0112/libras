from re import T
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np
from utils import Utils
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


aaaa = [True, False]

TRAIN = aaaa[0]

train = Utils.loadVideo('/home/thalles/Documents/PROJETO INTEGRADOR/DATABASE', ['oi', 'boa', 'tarde'])
x, y = Utils.processVideoMOD(train, 15, (300, 300))





x = np.array(x, 'float32')
x = x.reshape(450, 300,300,1)
y = np.array(y, 'float32')

print(x.shape)



if TRAIN:


    net= Sequential(
                    [
                     layers.Conv2D(32,(3,3),strides=(1,1),activation="relu",input_shape=(300,300,1)),
                     layers.MaxPooling2D(),
                     layers.Flatten(),
                     layers.Dense(64,activation="relu"),
                     layers.Dense(3,activation="softmax")
                    ]
                    )
    net.compile(optimizer='adam',
            loss='SparseCategoricalCrossentropy',
            metrics=['accuracy'])


    net.fit(x, y, 32, 7)

   # net.save('modelconv2d-3')