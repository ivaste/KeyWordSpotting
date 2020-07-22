import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import regularizers 

def SimpleNN(nCategories, inputShape, name="SimpleNN"):
	inputs = keras.Input(shape=inputShape)
	x = layers.Flatten()(inputs)
	x = layers.Dense(64,
					activation="elu",
					kernel_initializer="he_normal")(x)
	x = layers.Dense(64,
					activation="elu",
					kernel_initializer="he_normal")(x)
	x = layers.Dense(64,
					activation="elu",
					kernel_initializer="he_normal")(x)
	output = layers.Dense(nCategories,
						activation="softmax",
						kernel_initializer="glorot_uniform")(x)

	model = keras.Model(inputs=inputs, outputs=output, name=name)
	
	return model
	


def LeNet5(nCategories, inputShape, name="LeNet5-2FC-Reg"):
	inputs = keras.Input(shape=inputShape)
	# dim Input=(120, 126, 1)
	
	regu=regularizers.l2(1e-2)
	
	x = layers.Conv2D(filters=6,
					kernel_size=5,
					strides=(1,1),
					activation='elu',
					padding='valid',
					kernel_initializer="he_normal",
					kernel_regularizer=regu)(inputs)
	x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
	x = layers.Conv2D(filters=16,
					kernel_size=5,
					strides=(1,1),
					activation='elu',
					padding='valid',
					kernel_initializer="he_normal",
					kernel_regularizer=regu)(x)
	x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
	x = layers.Flatten()(x)
	x = layers.Dense(120,
					activation='elu',
					kernel_initializer="he_normal")(x)
	'''x=layers.Dense(84,
					activation='elu',
					kernel_initializer="he_normal")(x)'''
	output = layers.Dense(nCategories,
						activation="softmax",
						kernel_initializer="glorot_uniform",
						kernel_regularizer=regu)(x)
	
	model = keras.Model(inputs=inputs, outputs=output, name=name)
	
	return model



# From paper https://www.researchgate.net/publication/332553888_End-to-End_Environmental_Sound_Classification_using_a_1D_Convolutional_Neural_Network
def directCNN(nCategories,inputShape, name="directCNN"):
	inputs = keras.Input(shape=inputShape)
	# dim Input:(16000, 1)
	
	x = layers.Conv1D(filters=16,
					kernel_size=64,
					strides=2,
					activation='relu',
					padding='valid',
					kernel_initializer="he_normal")(inputs)
	# dim: (7969,16)
	x = layers.MaxPooling1D(pool_size=8, strides=8)(x)
	# dim: (996,16)
	x = layers.Conv1D(filters=32,
					kernel_size=32,
					strides=2,
					activation='relu',
					padding='valid',
					kernel_initializer="he_normal")(x)
	# dim: (483,32)
	x = layers.MaxPooling1D(pool_size=8, strides=8)(x)
	# dim: (60,32)
	x = layers.Conv1D(filters=64,
					kernel_size=16,
					strides=2,
					activation='relu',
					padding='valid',
					kernel_initializer="he_normal")(x)
	# dim: (23,64)
	x = layers.Conv1D(filters=128,
					kernel_size=8,
					strides=2,
					activation='relu',
					padding='valid',
					kernel_initializer="he_normal")(x)
	# dim: (8,128)
	
	x = layers.Flatten()(x)
	x=layers.Dense(128,
					activation='relu',
					kernel_initializer="he_normal")(x)
	x=layers.Dense(64,
					activation='relu',
					kernel_initializer="he_normal")(x)
	
	output = layers.Dense(nCategories,
						activation="softmax",
						kernel_initializer="glorot_uniform")(x)

	model = keras.Model(inputs=inputs, outputs=output, name=name)

	return model





def AttRNNSpeechModel(nCategories, inputShape, rnn_func=layers.LSTM, name="AttNN"):
    inputs = keras.Input(shape=inputShape)
	#x = layers.Flatten()(inputs)

    x = layers.Conv2D(10, 5, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = layers.Lambda(lambda q: backend.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = layers.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]
    x = layers.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]

    xFirst = layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = layers.Dense(128)(xFirst)

    # dot product attention
    attScores = layers.Dot(axes=[1, 2])([query, x])
    attScores = layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = layers.Dense(64, activation='relu')(attVector)
    x = layers.Dense(32)(x)

    output = layers.Dense(nCategories, activation='softmax', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=output, name=name)

    return model