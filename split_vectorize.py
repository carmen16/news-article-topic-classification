
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

class SplitVectorize:

	def __init__(self, df, articles, article_type, labels='desk', n_labels=20):
		self.labels = labels
		self.articles = articles
		#self.category_min = category_min
		self.name_ = article_type

		# Get list of top N labels
		label_count = df.groupby(self.labels, as_index=False).count().sort_values(by=['headline'], ascending=False).reset_index()[[self.labels, 'headline']].rename(columns={'headline':'cnt'})
		self.top_n_labels_ = set(label_count[self.labels][:n_labels])

		# Overwrite extra label values with 'other'
		cleaned_labels = df[self.labels].apply(lambda x: x if x in self.top_n_labels_ else '<OTHER>')
		self.df = pd.DataFrame(pd.concat([pd.DataFrame(cleaned_labels, columns={self.labels}),
			df[self.articles]], axis=1))
		label_count = self.df.groupby(self.labels, as_index=False).count().sort_values(by=[self.articles], ascending=False).reset_index()[[self.labels, self.articles]]
		self.n_classes_ = len(label_count[self.labels])

		plt.bar(range(self.n_classes_), label_count[self.articles], color='#1f77b4')
		#plt.title('Top '+str(n_labels)+' Categories + <OTHER>')
		plt.xticks(range(self.n_classes_), label_count[self.labels], rotation=45, horizontalalignment='right')
		#plt.xlabel(self.labels)
		plt.ylabel('Number of Articles')
		plt.tight_layout()
		plt.savefig('plots/top_n_labels_histogram.png')


	def train_test_split(self, rand_seed):
		# Remove null entries
		data = self.df[[self.articles, self.labels]][self.df[self.articles].notnull()][self.df[self.labels].notnull()]
		data = data.reset_index()[[self.articles, self.labels]]
		
		# Remove entries in categories with too few data points
		#data = data.groupby(self.labels).filter(lambda x: x[self.labels].count() >= self.category_min)
		#data = data.reset_index()[[self.articles, self.labels]]

		# Randomly assign indexes to 75% train, 5% dev, 20% test
		np.random.seed(rand_seed)
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


	def vocab_stats(self):
		# Count number of training words
		num_words = 0
		unq_words = 0
		vocab = set()

		start = time.time()

		for i in range(self.train_data.shape[0]):
			tokens = self.train_data[i].split()
			token_set = set(tokens)

			num_words += len(tokens)
			unq_words += len(token_set)
			vocab = vocab.union(token_set)

			if (i+1) % 1000 == 0:
				elapsed = time.time() - start
				hours, rem = divmod(elapsed, 3600)
				minutes, seconds = divmod(rem, 60)
				print('{}: Scanned {:,} articles {:0>2}:{:0>2}:{:0>2}'.format(self.name_, i+1, int(hours), int(minutes), int(seconds)))

		self.train_words_ = num_words
		self.train_avg_words_ = num_words / self.train_data.shape[0]
		self.train_vocab_size_ = len(vocab)
		self.train_avg_unq_words_ = unq_words / self.train_data.shape[0]


	def tfidf_vectorize(self, min_df=2):
		# Fit TFIDF on training data and apply transformation to all 3 data pieces
		tv = TfidfVectorizer(min_df=min_df)
		tv.fit(self.train_data)
		self.tv_train = tv.transform(self.train_data)
		self.tv_dev = tv.transform(self.dev_data)
		self.tv_test = tv.transform(self.test_data)

