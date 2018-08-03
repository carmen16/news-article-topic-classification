
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

class MultNB:

	def __init__(self, alpha=list(2. ** np.arange(-12, 3, 0.5))):
		self.alpha = alpha

	def test_models(self, sv_obj):
		# Takes SplitVectorize object and creates accuracy table
		accuracy = []
		for i in range(len(self.alpha)):
			mnb = MultinomialNB(alpha=self.alpha[i])
			mnb.fit(sv_obj.tv_train, sv_obj.train_labels)
			accuracy.append(mnb.score(sv_obj.tv_dev, sv_obj.dev_labels))
		
		# Combine to dataframe
		pd.options.display.float_format = '{:,.8f}'.format
		df = pd.concat([pd.DataFrame([sv_obj.name_]*len(self.alpha), columns={'input'}),
			pd.DataFrame(self.alpha, columns={'alpha'}),
			pd.DataFrame(accuracy, columns={'accuracy'})], axis=1)
		sv_obj.mnb_accuracy_table_ = df

		# Hold onto best model
		best_alpha = df[df.accuracy == max(df.accuracy)].alpha.iloc[0]

		sv_obj.best_mnb_model_ = MultinomialNB(alpha=best_alpha)
		sv_obj.best_mnb_model_.fit(sv_obj.tv_train, sv_obj.train_labels)

	def plot_accuracy(self, models, colors=['blue', 'red', 'green', 'orange', 'pink']):
		# Combine accuracy tables into one to find the most accurate model
		x = models[0]
		for i in range(len(models)-1):
			x = pd.concat([x, models[i+1]], axis=0)
		x = x.reset_index()[['input','alpha','accuracy']]
		self.combined_accuracy_ = x

		print('Best Multinomial Naïve Bayes model:')
		print(x.iloc[np.where(x.accuracy == max(x.accuracy))[0][0]])

		# Plot multinomial NB accuracies
		self.colors = colors
		log_alpha = np.log2(self.alpha)

		for i in range(len(models)):
			plt.plot(log_alpha, models[i].accuracy, color=self.colors[i], marker='o', label=models[i].input[0])

		plt.title('Multinomial Naïve Bayes on TFIDF Vectorizer')
		plt.xlabel('Log2(Alpha)')
		plt.ylabel('Accuracy on Dev Data')
		plt.ylim(0, 1)
		plt.legend()
		plt.savefig('plots/mnb_accuracy.png', orientation='landscape')


class LogReg:

	def __init__(self, C=list(10. ** np.arange(-7, 7)), penalties=['l1', 'l2']):
		self.C = C
		self.penalties = penalties

	def test_models(self, sv_obj):
		# Takes SplitVectorize object and creates accuracy table
		accuracy = []
		for i in self.penalties:
			for j in range(len(self.C)):
				lr = LogisticRegression(C=self.C[j], penalty=i)
				lr.fit(sv_obj.tv_train, sv_obj.train_labels)
				accuracy.append(lr.score(sv_obj.tv_dev, sv_obj.dev_labels))
	
		# Combine to dataframe
		pd.options.display.float_format = '{:,.8f}'.format
		df = pd.concat([pd.DataFrame([sv_obj.name_]*len(self.C)*2, columns={'input'}),
			pd.DataFrame(['L1']*len(self.C) + ['L2']*len(self.C), columns={'penalty'}),
			pd.DataFrame(self.C*len(self.penalties), columns={'C'}), pd.DataFrame(accuracy, columns={'accuracy'})], axis=1)
		sv_obj.lr_accuracy_table_ = df

		# Hold onto best model
		best_C = df[df.accuracy == max(df.accuracy)].C.iloc[0]
		best_penalty = df[df.accuracy == max(df.accuracy)].penalty.iloc[0].lower()

		sv_obj.best_lr_model_ = LogisticRegression(C=best_C, penalty=best_penalty)
		sv_obj.best_lr_model_.fit(sv_obj.tv_train, sv_obj.train_labels)

	def plot_accuracy(self, models, colors=['blue', 'red', 'green', 'orange', 'pink']):
		# Combine accuracy tables into one to find the most accurate model
		x = models[0]
		for i in range(len(models)-1):
			x = pd.concat([x, models[i+1]], axis=0)
		x = x.reset_index()[['input','penalty','C','accuracy']]
		self.combined_accuracy_ = x

		print('Best Logistic Regression model:')
		print(x.iloc[np.where(x.accuracy == max(x.accuracy))[0][0]])

		# Plot logistic regression accuracies
		self.colors = colors
		log_C = np.log10(self.C)

		for i in range(len(models)):
			plt.plot(log_C, models[i].accuracy[np.where(models[i].penalty == 'L1')[0]],
				color=self.colors[i], linestyle='dotted', marker='x', label=models[i].input[0]+'-L1')
			plt.plot(log_C, models[i].accuracy[np.where(models[i].penalty == 'L2')[0]],
				color=self.colors[i], marker='o', label=models[i].input[0]+'-L2')

		plt.title('Logistic Regression on TFIDF Vectorizer')
		plt.xlabel('Log10(C)')
		plt.ylabel('Accuracy on Dev Data')
		plt.ylim(0, 1)
		plt.legend()
		plt.savefig('plots/lr_accuracy.png', orientation='landscape')

