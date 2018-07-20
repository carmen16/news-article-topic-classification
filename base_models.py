
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

class MultNB:

	def __init__(self, alpha=list(2. ** np.arange(-12, 3, 0.5))):
		self.alpha = alpha

	def accuracy_table(self, sv_obj):
		# Takes SplitVectorize object and creates accuracy table
		accuracy = []
		for i in range(len(self.alpha)):
			mnb = MultinomialNB(alpha=self.alpha[i])
			mnb.fit(sv_obj.tv_train, sv_obj.train_labels)
			accuracy.append(mnb.score(sv_obj.tv_dev, sv_obj.dev_labels))
		
		# Combine to dataframe
		pd.options.display.float_format = '{:,.5f}'.format
		df = pd.concat([pd.DataFrame([sv_obj.name]*len(self.alpha), columns={'name'}),
			pd.DataFrame(self.alpha, columns={'alpha'}),
			pd.DataFrame(accuracy, columns={'accuracy'})], axis=1)
		return df

	def plot_accuracy(self, models, colors=['blue', 'red', 'green', 'orange']):
		# Combine accuracy tables into one to find the most accurate model
		x = models[0]
		for i in range(len(models)-1):
			x = pd.concat([x, models[i+1]], axis=0)
		x = x.reset_index()[['name','alpha','accuracy']]
		self.combined_accuracy_ = x

		print('Best Multinomial Naïve Bayes model:')
		print(x.iloc[np.where(x.accuracy == max(x.accuracy))[0][0]])

		# Plot multinomial NB accuracies
		self.colors = colors
		log_alpha = np.log2(self.alpha)

		for i in range(len(models)):
			plt.plot(log_alpha, models[i].accuracy, color=self.colors[i], marker='o', label=models[i].name[0])

		plt.title('Multinomial Naïve Bayes on TFIDF Vectorizer')
		plt.xlabel('Log2(Alpha)')
		plt.ylabel('Accuracy')
		plt.ylim(0, 1)
		plt.legend()
		plt.savefig('mnb_accuracy.png', orientation='landscape')


class LogReg:

	def __init__(self, C=list(10. ** np.arange(-8, 8)), penalties=['l1', 'l2']):
		self.C = C
		self.penalties = penalties

	def accuracy_table(self, sv_obj):
		# Takes SplitVectorize object and creates accuracy table
		accuracy = []
		for i in self.penalties:
			for j in range(len(self.C)):
				lr = LogisticRegression(C=self.C[j], penalty=i)
				lr.fit(sv_obj.tv_train, sv_obj.train_labels)
				accuracy.append(lr.score(sv_obj.tv_dev, sv_obj.dev_labels))
	
		# Combine to dataframe
		pd.options.display.float_format = '{:,.8f}'.format
		df = pd.concat([pd.DataFrame([sv_obj.name]*len(self.C)*2, columns={'name'}),
			pd.DataFrame(['L1']*len(self.C) + ['L2']*len(self.C), columns={'penalty'}),
			pd.DataFrame(self.C*len(self.penalties), columns={'C'}), pd.DataFrame(accuracy, columns={'accuracy'})], axis=1)
		return df


	def plot_accuracy(self, models, colors=['blue', 'red', 'green', 'orange']):
		# Combine accuracy tables into one to find the most accurate model
		x = models[0]
		for i in range(len(models)-1):
			x = pd.concat([x, models[i+1]], axis=0)
		x = x.reset_index()[['name','penalty','C','accuracy']]
		self.combined_accuracy_ = x

		print('Best Logistic Regression model:')
		print(x.iloc[np.where(x.accuracy == max(x.accuracy))[0][0]])

		# Plot logistic regression accuracies
		self.colors = colors
		log_C = np.log10(self.C)

		for i in range(len(models)):
			plt.plot(log_C, models[i].accuracy[np.where(models[i].penalty == 'L1')[0]],
				color=self.colors[i], linestyle='dotted', marker='x', label=models[i].name[0]+'-L1')
			plt.plot(log_C, models[i].accuracy[np.where(models[i].penalty == 'L2')[0]],
				color=self.colors[i], marker='o', label=models[i].name[0]+'-L2')

		plt.title('Logistic Regression on TFIDF Vectorizer')
		plt.xlabel('Log10(C)')
		plt.ylabel('Accuracy')
		plt.ylim(0, 1)
		plt.legend()
		plt.savefig('lr_accuracy.png', orientation='landscape')

