import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import pickle
import numpy as np
from utils import Utils







TRAIN = True

traindata = Utils.loadVideo('/home/thalles/Documents/PROJETO INTEGRADOR/DATABASE', ['oi', 'boa', 'tarde'])

x, labels = Utils.processVideoMOD(traindata, size=(300,300))


x = np.array(x, 'uint8')
y = np.array(labels, 'uint8')

x = np.expand_dims(x, 0)
y = np.expand_dims(y, 0)

print(y.shape)
print(x.shape)


if TRAIN:
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        
        layers.TimeDistributed(layers.Flatten(input_shape=(300,300))),
        layers.TimeDistributed(layers.Dense(128, 'relu')),
        layers.TimeDistributed(layers.Dense(512, 'sigmoid')),
        layers.TimeDistributed(layers.Dense(240, 'relu')),
       layers.TimeDistributed(layers.Dense(len(labels), 'relu'))
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'], )

    model.fit(x, y, epochs=14)

    test_loss, test_Acc = model.evaluate(x, y, verbose=5)

    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])


    predictions = np.argmax(probability_model.predict(x))

    #probability_model.save('data/modelLibras-tests')

print(x.shape)