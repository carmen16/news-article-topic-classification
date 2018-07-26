
import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

class SplitVectorize:

	def __init__(self, df, articles, article_type, labels='desk', category_min=10):
		self.df = df
		self.articles = articles
		self.labels = labels
		self.category_min = category_min
		self.name_ = article_type

	def train_test_split(self):
		# Remove null entries
		data = self.df[[self.articles, self.labels]][self.df[self.articles].notnull()][self.df[self.labels].notnull()]
		data = data.reset_index()[[self.articles, self.labels]]
		
		# Remove entries in categories with too few data points
		data = data.groupby(self.labels).filter(lambda x: x[self.labels].count() >= self.category_min)
		data = data.reset_index()[[self.articles, self.labels]]

		# Randomly assign indexes to 75% train, 5% dev, 20% test
		np.random.seed(102)
		rand = np.random.uniform(size=len(data[self.labels]))
		train_idx = np.where(rand < 0.75)[0]
		dev_idx = np.where((rand >= 0.75) & (rand < 0.8))[0]
		test_idx = np.where(rand >= 0.8)[0]
		
		# Break dataframe into train, dev, test dfs
		train = data.loc[list(train_idx)].reset_index()[[self.articles, self.labels]]
		dev = data.loc[list(dev_idx)].reset_index()[[self.articles, self.labels]]
		test = data.loc[list(test_idx)].reset_index()[[self.articles, self.labels]]

		self.train_data, self.train_labels = train[self.articles], train[self.labels]
		self.dev_data, self.dev_labels = dev[self.articles], dev[self.labels]
		self.test_data, self.test_labels = test[self.articles], test[self.labels]

		# Count number of training words
		tokenizer = TreebankWordTokenizer()
		num_words = 0

		for i in range(len(self.train_data)):
			num_words += len(tokenizer.tokenize(self.train_data[i]))
		self.train_words_ = num_words

	def tfidf_vectorize(self):
		# Fit TFIDF on training data and apply transformation to all 3 data pieces
		tv = TfidfVectorizer()
		tv.fit(self.train_data)
		self.tv_train = tv.transform(self.train_data)
		self.tv_dev = tv.transform(self.dev_data)
		self.tv_test = tv.transform(self.test_data)

		self.train_vocab_size_ = self.tv_train.shape[1]
		self.train_avg_unq_words_ = self.tv_train.nnz / self.tv_train.shape[0]
