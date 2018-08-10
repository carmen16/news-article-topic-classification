import numpy as np
import pandas as pd
import tensorflow as tf

def load_glove_vectors(embed_dim):
	# Convert txt file into dict with words as keys and list(float) as values
	filename = 'GloVe/glove.6B.'+str(embed_dim)+'d.txt'
	embed_index = {}

	with open(filename) as f:
		for line in f:
			line_s = line.split()
			word = line_s[0]
			vector = np.asarray(line_s[1:], dtype='float32')
			embed_index[word] = vector

	return embed_index


class DataPreparation:

	def __init__(self):
		pass


	def tokenize(self, sv_obj):
		tokens = sv_obj.train_data.str.split()
		self.vocab = sv_obj.tv_train.shape[1] #int(round(sv_obj.train_vocab_size_ * 0.8, 0))
		print('{}:\n\tTraining vocab size: {:,}'.format(sv_obj.name_, self.vocab))

		# Convert words to IDs
		t = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab, oov_token='<unk>')
		t.fit_on_texts(tokens)#sv_obj.train_data)
		self.train = t.texts_to_sequences(sv_obj.train_data)
		self.test = t.texts_to_sequences(sv_obj.test_data)

		# Save index of words to IDs
		sv_obj.word_index = t.word_index

		# Save index of IDs to words
		sv_obj.id_index = {}
		for key, val in sv_obj.word_index.items():
			sv_obj.id_index[val] = key


	def pad(self, sv_obj):
		# Gather length distribution for training data
		lens = []
		for i in range(len(self.train)):
			lens.append(len(self.train[i]))

		# Pad word IDs to min(90% of max article length, 500)
		pct = int(round(np.percentile(lens, 90), 0))
		sv_obj.maxlen = min(pct, 500)
		print('\t90th percentile of length = {:,} --> inputs padded to {:,}.'.format(pct, sv_obj.maxlen))
		train_pad = tf.keras.preprocessing.sequence.pad_sequences(self.train, maxlen=sv_obj.maxlen, padding='post', truncating='post')
		test_pad = tf.keras.preprocessing.sequence.pad_sequences(self.test, maxlen=sv_obj.maxlen, padding='post', truncating='post')

		# Check new vocab size
		s = set()
		for i in range(train_pad.shape[0]):
			s = s.union(set(train_pad[i]))
		sv_obj.nn_effective_vocab_ = len(s)
		print('\tVocab reduced to {:,} by padding operation ({:.0%})'.format(len(s), (len(s)/self.vocab)-1))

		# Attach ID sequences to SV object
		sv_obj.train_ids = train_pad
		sv_obj.test_ids = test_pad

		print('\tTraining data shape: {}'.format(sv_obj.train_ids.shape))


	def create_glove_matrix(self, sv_obj, embed_dim=100):
		sv_obj.nn_embed_dim_ = embed_dim

		embed_index = load_glove_vectors(embed_dim)

		# Create matrix of GloVe weights
		sv_obj.embed_matrix = np.zeros((len(sv_obj.word_index) + 1, embed_dim))
		for word, i in sv_obj.word_index.items():
			embed_vector = embed_index.get(word)
			if embed_vector is not None:
				# Words not found in embedding index will be all zeros.
				sv_obj.embed_matrix[i] = embed_vector

		print('\tCreated GloVe matrix of {:,}-dimensional embeddings'.format(embed_dim))


class CNN:

	def cnn(self, sv_obj, num_layers=3, filters=[4, 16, 32], kernel_size=[20, 10, 5], epochs=5):
		train_data = sv_obj.train_ids
		train_labels = sv_obj.train_labels.apply(lambda x: sv_obj.label_ids[x]) #pd.factorize(sv_obj.train_labels)[0]
		test_data = sv_obj.test_ids
		test_labels = sv_obj.test_labels.apply(lambda x: sv_obj.label_ids[x]) #pd.factorize(sv_obj.test_labels)[0]
		n_classes = sv_obj.n_classes_

		model = tf.keras.Sequential()

		# Add embedding layer
		embedding_layer = tf.keras.layers.Embedding(len(sv_obj.word_index) + 1, sv_obj.nn_embed_dim_,
			weights=[sv_obj.embed_matrix],
			input_length=sv_obj.maxlen)#,
			#trainable=False)
		model.add(embedding_layer)

		# Add convolution layers
		for i in range(num_layers-1):
			model.add(tf.keras.layers.Conv1D(filters=filters[i],
				kernel_size=kernel_size[i],
				activation=tf.nn.relu))
			#print('Conv1D:', model.output_shape)

		# Add pooling layers
		max_pooling_layer = tf.keras.layers.MaxPooling1D()
		model.add(max_pooling_layer)
		#print('MaxPooling:', model.output_shape)

		model.add(tf.keras.layers.Conv1D(filters=filters[num_layers-1],
			kernel_size=kernel_size[num_layers-1],
			activation=tf.nn.relu))
		#print('Conv1D:', model.output_shape)

		avg_pooling_layer = tf.keras.layers.GlobalAveragePooling1D()
		model.add(avg_pooling_layer)
		#print('AvgPooling:', model.output_shape)

		# Add fully connected layers
		model.add(tf.keras.layers.Dense(sv_obj.nn_embed_dim_, activation=tf.nn.relu))
		#print('ReLU:', model.output_shape)
		model.add(tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax))
		#print('Softmax:', model.output_shape)

		# Compile and fit model
		print('\n'+sv_obj.name_.upper()+':\n')
		model.compile(optimizer=tf.train.AdamOptimizer(),
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

		model.fit(train_data, train_labels, epochs=epochs)
		sv_obj.cnn_model_ = model

		test_loss, test_acc = model.evaluate(test_data, test_labels)
		print('Test accuracy:', test_acc)
		sv_obj.cnn_accuracy_ = test_acc

		#pred = model.predict(test_data)

		print(model.summary())

