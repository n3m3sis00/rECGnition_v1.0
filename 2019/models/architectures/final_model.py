from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

class AttentionModule(layers.Layer):

	def __init__(self,**kwargs):
		super(AttentionModule, self).__init__(**kwargs)

	def build(self, input_shape):
		#self.kernel = layers.Conv2D(1, (input_shape[-1], input_shape[-2]), use_bias=False, kernel_initializer="ones")
		self.kernel = self.add_weight(shape=[64,64,3], initializer="ones")

	def call(self, input_tensor):
		attended_weights = tf.nn.softmax(tf.nn.relu(tf.multiply(self.kernel, input_tensor)))
		#repeat_ = layers.RepeatVector(3)(attended_weights)
		attended_input = tf.multiply(attended_weights,input_tensor)
		return attended_input

def getModel():
    input1 = keras.Input(shape=(2,), name='age_gen')
    x = layers.Dense(64, activation='relu')(input1)
    x = layers.Dense(64, activation='relu')(x)


    input2 = keras.Input(shape=(64, 64, 3), name='img')
    y = AttentionModule()(input2)
    y = layers.Conv2D(32, 3, activation='relu')(y)
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
    output = layers.Dense(14, activation='softmax')(combined)

    model = keras.Model(inputs=[input1, input2], outputs=output)

    return model

