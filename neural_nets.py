import numpy as np
import pandas as pd
import tensorflow as tf

class NeuralNets:

	def __init__(self, embed_dim=100): #load_glove_vectors
		self.embed_dim = embed_dim
		# Convert txt file into dict with words as keys and list(float) as values
		filename = 'GloVe/glove.6B.'+str(self.embed_dim)+'d.txt'
		self.embed_index = {}

		with open(filename) as f:
			for line in f:
				line_s = line.split()
				word = line_s[0]
				vector = np.asarray(line_s[1:], dtype='float32')
				self.embed_index[word] = vector

		print('Created index of {:,}-dimensional embeddings for {:,} words.'.format(self.embed_dim, len(self.embed_index)))


	def tokenize_pad(self, sv_obj):
		# Convert words to IDs
		t = tf.keras.preprocessing.text.Tokenizer()
		t.fit_on_texts(sv_obj.train_data)
		train = t.texts_to_sequences(sv_obj.train_data)
		test = t.texts_to_sequences(sv_obj.test_data)

		# Save index of words to IDs and create matrix of GloVe weights
		sv_obj.word_index = t.word_index
		print('{}:\n\t{:,} unique tokens'.format(sv_obj.name_, len(sv_obj.word_index)))

		sv_obj.embed_matrix = np.zeros((len(sv_obj.word_index) + 1, self.embed_dim))
		for word, i in sv_obj.word_index.items():
			embed_vector = self.embed_index.get(word)
			if embed_vector is not None:
				# Words not found in embedding index will be all zeros.
				sv_obj.embed_matrix[i] = embed_vector
		print('\tCreated matrix of {:,}-dimensional weights for {:,} tokens'.format(sv_obj.embed_matrix.shape[1], sv_obj.embed_matrix.shape[0]))

		# Gather some statistics on training data
		#words = 0
		lens = []
		#vocab = []
		#unq = []
		for i in range(len(train)):
		#	words += len(train[i])
			lens.append(len(train[i]))
		#	vocab.append(max(train[i]))
		#	unq.append(len(set(train[i])))

		#sv_obj.train_words_ = words
		#sv_obj.train_avg_words_ = sv_obj.train_words_ / len(train)
		#sv_obj.train_vocab_size_ = max(vocab)
		#sv_obj.train_avg_unq_words_ = sum(unq) / len(train)

		# Pad word IDs to 90% of max article length
		sv_obj.maxlen = int(round(np.percentile(lens, 90), 0))
		train_pad = tf.keras.preprocessing.sequence.pad_sequences(train, maxlen=sv_obj.maxlen, padding='post', truncating='post')
		test_pad = tf.keras.preprocessing.sequence.pad_sequences(test, maxlen=sv_obj.maxlen, padding='post', truncating='post')

		# Attach ID sequences to SV object
		sv_obj.train_ids = train_pad
		sv_obj.test_ids = test_pad

		print('\tTraining data shape: {}'.format(sv_obj.train_ids.shape))


	def neural_net(self, sv_obj, epochs=5):
		train_data = sv_obj.train_ids
		train_labels = pd.factorize(sv_obj.train_labels)[0]
		test_data = sv_obj.test_ids
		test_labels = pd.factorize(sv_obj.test_labels)[0]
		n_classes = sv_obj.n_classes

		embedding_layer = tf.keras.layers.Embedding(len(sv_obj.word_index) + 1, self.embed_dim,
			weights=[sv_obj.embed_matrix],
			input_length=sv_obj.maxlen,
			trainable=False)

		model = tf.keras.Sequential([
			embedding_layer,
			tf.keras.layers.GlobalAveragePooling1D(),
			tf.keras.layers.Dense(self.embed_dim, activation=tf.nn.relu),
			tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
		])

		model.compile(optimizer=tf.train.AdamOptimizer(),
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])
		print('Compiled')

		model.fit(train_data, train_labels, epochs=epochs)
		print('Fit')

		test_loss, test_acc = model.evaluate(test_data, test_labels)
		print('Test loss:', test_loss)
		print('Test accuracy:', test_acc)

		pred = model.predict(test_data)
		print('First prediction (probabilities):', pred[0])
		print('First prediction (category):', np.argmax(pred[0]))
		print('First test label:', test_labels[0])

		print(model.summary())

		return model


#def imdb_nn(sv_obj):