from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


def getModel():
    input1 = keras.Input(shape=(2,), name='age_gen')
    x = layers.Dense(64, activation='relu')(input1)
    x = layers.Dense(64, activation='relu')(x)
    
    input2 = keras.Input(shape=(64, 64, 3), name='img')
    y = layers.Conv2D(32, 3, activation='relu')(input2)
    y = layers.Dropout(0.1)(y)
    y = layers.Conv2D(32, 3, activation='relu')(y)
    y = layers.MaxPooling2D(2)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(64, 3, activation='relu')(y)
    y = layers.MaxPooling2D(2)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(64, 3, activation='relu')(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Conv2D(128, 3, activation='relu')(y)
    y = layers.MaxPooling2D(2)(y)
    y = layers.BatchNormalization()(y)

    
    y = layers.Flatten()(y)


    combined = layers.concatenate([x, y])
    combined = layers.Dense(64, activation='sigmoid')(combined)
    y = layers.Dropout(0.2)(y)
    output = layers.Dense(2, activation='softmax')(combined)

    model = keras.Model(inputs=[input1, input2], outputs=output)

    return model

