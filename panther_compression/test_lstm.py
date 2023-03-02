### test for LSTM

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # suppress tensorflow msgs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


if __name__=="__main__":

	##
	## params
	##

	use_saved_lstm = str(sys.argv[1]) == "True"

	##
	## data
	##

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.astype("float32") / 255.0
	x_test = x_test.astype("float32") / 255.0

	if not use_saved_lstm:

		# physical_devices = tf.config.list_physical_devices("GPU")
		# tf.config.experimental.set_memory_growth(physical_devices[0], True)

		model = keras.Sequential()
		# model.add(keras.Input(shape=(None, 28)))
		model.add(layers.LSTM(256, return_sequences=True, activation='relu')) # first LSTM layer
		model.add(layers.LSTM(256, activation='relu')) # second LSTM layer
		model.add(layers.Dense(10))

		model.compile(
			loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			optimizer=keras.optimizers.Adam(learning_rate=0.001),
			metrics=["accuracy"]
		)

		model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=2)
		model.save('./trained_policies/lstm/lstm')

	else:
		model = keras.models.load_model('./trained_policies/lstm/lstm')

	print("evaluate!")
	for i in range(28, 20, -1):
		print(f"i={i}: ")
		model.evaluate(x_test[:5000,:i,:], y_test[:5000], batch_size=64, verbose=2)
