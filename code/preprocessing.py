import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordTokenizer

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
	return wn_tags
	
def create_nouns_lemmas(data):
	# Given a list or column of articles, returns a nouns-only version and a
	# lemmatized version of the articles
	tokenizer = TreebankWordTokenizer()
	lemmatizer = WordNetLemmatizer()

	nouns = []
	lemmas = []

	for i in range(len(data)):
		if pd.isnull(data[i]):
			noun_string = np.nan
		else:
			tokens = tokenizer.tokenize(data[i])
			tagged_tokens = pos_tag(tokens)
			wn_tags = treebank_to_wn_tags(tagged_tokens)
			
			# For nouns
			noun_list = [x[0] for x in wn_tags if x[1] == 'n']
			noun_string = ' '.join(noun_list)
			
			# For lemmas
			lemma_list = [lemmatizer.lemmatize(x[0], pos=x[1]) for x in wn_tags]
			lemma_string = ' '.join(lemma_list)
		
		nouns.append(noun_string)
		lemmas.append(lemma_string)

	return pd.DataFrame(nouns, columns={'nouns'}), pd.DataFrame(lemmas, columns={'lemmas'})


def clean_labels(c):
	# Regex operations to clean labels and collapse some categories
    c = c.str.lower().str.strip()
    c = c.str.replace('desk', '')
    c = c.str.replace(';', '')
    c = c.str.replace(' and ', ' & ')
    c = c.str.replace('\\', '/')
    c = c.str.replace('arts & .*|cultural|museums|the arts/cultural|.*weekend.*', 'arts')
    c = c.str.replace('automobiles', 'cars')
    c = c.str.replace('classifed|classifieds|job market', 'classified')
    c = c.str.replace('.*dining out.*', 'dining')
    c = c.str.replace('education.*', 'education')
    c = c.str.replace('business/financ.*|business world magazine|e-commerce|.*money.*financ.*|sundaybusiness', 'business')
    c = c.str.replace('health&fitness', 'health & fitness')
    c = c.str.replace('home|house & home/style', 'home & garden')
    c = c.str.replace('metropolitian', 'metropolitan')
    c = c.str.replace('new jersey.*', 'new jersey weekly')
    c = c.str.replace('connecticut weekly|new jersey weekly|long island weekly|the city weekly.*|westchester weekly', 'city & region weekly')
    c = c.str.replace('thursday styles|styles of the times', 'style')
    c = c.str.replace('.*design.*magazine|.*fashion.*magazine|.*style.*magazine|.*travel.*magazine|t: \w+.*', 't magazine')
    c = c.str.replace('adventure sports|sports sports', 'sports')
    c = c.str.replace('circuits|flight', 'technology')
    c = c.str.strip()
    return c


def save_data(df, filename):
	df.to_csv(filename, index = False)


def process_data():
	df = pd.read_csv('../data/nyt_corpus.csv')
	print('Imported file')

	nouns, lemmas = create_nouns_lemmas(df.full_text)
	print('Created nouns and lemmas')

	labels = pd.DataFrame(clean_labels(df.desk))
	print('Cleaned labels')
	
	df_final = pd.DataFrame(pd.concat([labels, df.full_text, df.lead_paragraph,
		pd.DataFrame(nouns, columns={'nouns'}),
		pd.DataFrame(lemmas, columns={'lemmas'})], axis=1))

	save_data(df_final, '../data/nyt_corpus_cleaned.csv')
	print('Saved file')


if __name__ == '__main__':
	process_data()

