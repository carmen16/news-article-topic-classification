import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def basic_nn(sv_obj, embed_dim=100, epochs=5):
	train_data = sv_obj.tv_train.toarray()
	train_labels = pd.factorize(sv_obj.train_labels)[0]
	test_data = sv_obj.tv_test.toarray()
	test_labels = pd.factorize(sv_obj.test_labels)[0]
	num_classes = sv_obj.n_labels + 1

	model = keras.Sequential([
		keras.layers.Dense(embed_dim, activation=tf.nn.relu),
		keras.layers.Dense(num_classes, activation=tf.nn.softmax)
	])

	model.compile(optimizer=tf.train.AdamOptimizer(),
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])

	model.fit(train_data, train_labels, epochs=epochs)

	test_loss, test_acc = model.evaluate(test_data, test_labels)
	print('Test loss:', test_loss)
	print('Test accuracy:', test_acc)

	pred = model.predict(test_data)
	print('First prediction (probabilities):', pred[0])
	print('First prediction (category):', np.argmax(pred[0]))
	print('First test label:', test_labels[0])

	return model