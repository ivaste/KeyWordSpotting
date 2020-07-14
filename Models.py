import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def SimpleNN(nCategories, inputShape, name="SimpleNN"):
	inputs = keras.Input(shape=inputShape)
	x = layers.Flatten()(inputs)
	x = layers.Dense(64, activation="selu")(x)
	x = layers.Dense(32, activation="selu")(x)
	output = layers.Dense(nCategories, activation="softmax")(x)

	model = keras.Model(inputs=inputs, outputs=output, name=name)
	
	return model