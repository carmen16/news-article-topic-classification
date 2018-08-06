import pandas as pd
import numpy as np
import re
import sys
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordTokenizer

def clean_input_column(c):
	# Lower case everything
	c = c.str.lower().str.strip()

	# Separate punctuation from words
	outer_list = []
	punc = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
	for i in range(len(c)):
		if pd.isnull(c[i]):
			outer_list.append(np.nan)
		else:
			tokenizer = TreebankWordTokenizer()
			tokens = tokenizer.tokenize(c[i])

			inner_list = []
			for j in tokens:
				for k in punc:
					word_str = '([A-Za-z])'
					punc_str = '('+str('\\')+k+')'
					s1 = re.search(word_str+punc_str, j)
					if s1 is not None:
						j = re.sub(word_str+punc_str, s1.group(1)+' '+s1.group(2), j)
					s2 = re.search(punc_str+word_str, j)
					if s2 is not None:
						j = re.sub(punc_str+word_str, s2.group(1)+' '+s2.group(2), j)
				inner_list.extend(j.split())
			outer_list.append(' '.join(inner_list))

		if (i+1) % 1000 == 0:
			print('Cleaned '+str(i+1)+' entries in '+c.name)

	c = pd.DataFrame({c.name: outer_list})

	return c


def treebank_to_wn_tags(tagged_tokens):
	# Replaces Penn Treebank POS tags with WordNet POS tags
	wn_tags = []
	for x in tagged_tokens:
		if x[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
			wn_tags.append((x[0], wn.NOUN))
		elif x[1] in ['JJ', 'JJR', 'JJS']:
			wn_tags.append((x[0], wn.ADJ))
		elif x[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
			wn_tags.append((x[0], wn.VERB))
		elif x[1] in ['RB', 'RBR', 'RBS']:
			wn_tags.append((x[0], wn.ADV))
		else:
			wn_tags.append((x[0], wn.NOUN)) # catch-all for determinants, prepositions, etc.
	return wn_tags

	
def create_nouns_lemmas(data):
	# Given a list or column of articles, returns a nouns-only version and a
	# lemmatized version of the articles
	lemmatizer = WordNetLemmatizer()

	nouns = []
	lemmas = []

	for i in range(len(data)):
		if pd.isnull(data[i]):
			noun_string = np.nan
			lemma_string = np.nan
		else:
			tagged_tokens = pos_tag(data[i].split())
			
			# For nouns
			noun_list = [x[0] for x in tagged_tokens if x[1][0] == 'N']
			noun_string = ' '.join(noun_list)
			
			# For lemmas
			wn_tags = treebank_to_wn_tags(tagged_tokens)
			lemma_list = [lemmatizer.lemmatize(x[0], pos=x[1]) for x in wn_tags]
			lemma_string = ' '.join(lemma_list)
		
		nouns.append(noun_string)
		lemmas.append(lemma_string)

		if i > 0 and (i+1) % 1000 == 0:
			print('Created nouns and lemmas for '+str(i+1)+' articles')

	return pd.DataFrame({'nouns': nouns}), pd.DataFrame({'lemmas': lemmas})


def clean_labels(c):
	# Regex operations to clean labels and collapse some categories
	c = c.str.lower().str.strip()
	c = c.str.replace('desk', '')
	c = c.str.replace(';', '')
	c = c.str.replace(' and ', ' & ')
	c = c.str.replace('\\', '/')
	c = c.str.replace('arts & .*|cultural.*|museums|the arts/cultural|.*weekend.*', 'arts & leisure')
	c = c.str.replace('automobiles|automobile|automoblies', 'cars')
	c = c.str.replace('classifed|classifieds|classsified|classfied', 'classified')
	c = c.str.replace('workplace|working|retirement', 'job market')
	c = c.str.replace('book review dest', 'book review')
	c = c.str.replace('.*dining out.*', 'dining')
	c = c.str.replace('.*education.*', 'education')
	c = c.str.replace('business/financ.*|business world magazine|e-commerce|e-business|.*money.*financ.*|financial.*|small business|the business of green|sundaybusiness|^business $', 'business & financial')
	c = c.str.replace('health&fitness|the good health magazine|women\'s health|men & health', 'health & fitness')
	c = c.str.replace('circuitscircuits|circuits|flight|wireless living', 'technology')
	c = c.str.replace('home entertaining magazine|house & home/style|living living|living|home home|home', 'home & garden')
	c = c.str.replace('metropolitan dsk|metropolitian|qmetropolitan', 'metropolitan')
	c = c.str.replace('new jersey.*', 'new jersey weekly')
	c = c.str.replace('thursday styles|styles of the times|style of the times|tholiday', 'style')
	c = c.str.replace('.*design.*magazine|.*fashion.*magazine|.*style.*magazine|.*travel.*magazine|t: \w+.*', 't magazine')
	c = c.str.replace('entertaining magazine|new york, new york magazine|the new season magazine|world of new york magazine', 'magazine')
	c = c.str.replace('adventure sports|sports sports', 'sports')
	c = c.str.replace('millenium', 'millennium')
	c = c.str.replace('week in review.*|weekin review.*', 'week in review')
	c = c.str.replace('the city weekly.*|city weekly|the city.*', 'the city weekly')
	c = c.str.replace('escapes|vacation', 'travel')
	c = c.str.replace('.*real estate report', 'real estate')
	c = c.str.replace('.*supplement.*', 'supplement')
	c = c.str.strip()
	return c

def save_data(df, filename):
	df.to_csv(filename, index = False)


def process_data(n):
	df = pd.read_csv('data/nyt_corpus_'+str(n)+'.csv')
	print('Imported data file')

	full_text = clean_input_column(df.full_text)
	print('Cleaned full_text')
	lead_paragraph = clean_input_column(df.lead_paragraph)
	print('Cleaned lead_paragraph')
	headline = clean_input_column(df.headline)
	print('Cleaned headlines')

	nouns, lemmas = create_nouns_lemmas(full_text.full_text)
	print('Created nouns and lemmas')

	labels = pd.DataFrame(clean_labels(df.desk))
	print('Cleaned labels')
	
	df_final = pd.DataFrame(pd.concat([labels,
		full_text, lead_paragraph, headline, nouns, lemmas], axis=1))

	save_data(df_final, 'data/nyt_corpus_cleaned_'+str(n)+'.csv')
	print('Saved cleaned data file')


if __name__ == '__main__':
	n = int(sys.argv[1])
	process_data(n)

