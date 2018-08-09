import pandas as pd
import numpy as np
import re
import csv
import sys
import time
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordTokenizer


def strip_lower(s):
	return s.lower().strip()


def separate_punctuation(s):
	punc = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
	for p in punc:
		s = s.replace(p, ' '+p+' ')
	return s


def clean_input_column(s):
	if pd.isnull(s):
		return np.nan
	else:
		s = strip_lower(s)
		s = separate_punctuation(s)
		if len(s) > 32700:
			s = s[:32700]
		#tokenizer = TreebankWordTokenizer()
		#tokens = tokenizer.tokenize(s)

		#cleaned_list = []
		#for t in tokens:
		#	for p in punc:
		#		word_str = '([A-Za-z])'
		#		punc_str = '('+str('\\')+p+')'
		#		s1 = re.search(word_str+punc_str, t)
		#		if s1 is not None:
		#			t = re.sub(word_str+punc_str, s1.group(1)+' '+s1.group(2), t)
		#		s2 = re.search(punc_str+word_str, t)
		#		if s2 is not None:
		#			t = re.sub(punc_str+word_str, s2.group(1)+' '+s2.group(2), t)
		#	cleaned_list.extend(t.split())
		#s = ' '.join(cleaned_list)

	return s


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


def create_nouns_lemmas(s):
	# Given a string of the full text from an article, returns a nouns-only version
	# and a lemmatized version of the article
	tokenizer = TreebankWordTokenizer()
	lemmatizer = WordNetLemmatizer()

	if pd.isnull(s):
		nouns = np.nan
		lemmas = np.nan
	else:
		# Get rid of punctuation
		s = strip_lower(s)
		s = separate_punctuation(s)

		tokens = tokenizer.tokenize(s)
		tagged_tokens = pos_tag(tokens)
		
		# For nouns
		noun_list = [x[0] for x in tagged_tokens if x[1][0] == 'N']
		nouns = ' '.join(noun_list)
		
		# For lemmas
		wn_tags = treebank_to_wn_tags(tagged_tokens)
		lemma_list = [lemmatizer.lemmatize(x[0], pos=x[1]) for x in wn_tags]
		lemmas = ' '.join(lemma_list)

	return nouns, lemmas


def clean_label(s):
	# Regex operations to clean labels and collapse some categories
	s = strip_lower(s)
	s = s.replace('desk', '')
	s = s.replace(';', '')
	s = s.replace(' and ', ' & ')
	s = s.replace('\\', '/')
	s = re.sub('arts & .*|cultural.*|museums|the arts/cultural|.*weekend.*', 'arts & leisure', s)
	s = re.sub('automobiles|automobile|automoblies', 'cars', s)
	s = re.sub('classifed|classifieds|classsified|classfied', 'classified', s)
	s = re.sub('workplace|working|retirement', 'job market', s)
	s = re.sub('book review dest', 'book review', s)
	s = re.sub('.*dining out.*', 'dining', s)
	s = re.sub('.*education.*', 'education', s)
	s = re.sub('business/financ.*|business world magazine|e-commerce|e-business|.*money.*financ.*|financial.*|small business|the business of green|sundaybusiness|^business $', 'business & financial', s)
	s = re.sub('health&fitness|the good health magazine|women\'s health|men & health', 'health & fitness', s)
	s = re.sub('circuitscircuits|circuits|flight|wireless living', 'technology', s)
	s = re.sub('home entertaining magazine|house & home/style|living living|living|home home|home', 'home & garden', s)
	s = re.sub('metropolitan dsk|metropolitian|qmetropolitan', 'metropolitan', s)
	s = re.sub('new jersey.*', 'new jersey weekly', s)
	s = re.sub('thursday styles|styles of the times|style of the times|tholiday', 'style', s)
	s = re.sub('.*design.*magazine|.*fashion.*magazine|.*style.*magazine|.*travel.*magazine|t: \w+.*', 't magazine', s)
	s = re.sub('entertaining magazine|new york, new york magazine|the new season magazine|world of new york magazine', 'magazine', s)
	s = re.sub('adventure sports|sports sports', 'sports', s)
	s = re.sub('millenium', 'millennium', s)
	s = re.sub('week in review.*|weekin review.*', 'week in review', s)
	s = re.sub('the city weekly.*|city weekly|the city.*', 'the city weekly', s)
	s = re.sub('escapes|vacation', 'travel', s)
	s = re.sub('.*real estate report', 'real estate', s)
	s = re.sub('.*supplement.*', 'supplement', s)
	s = s.strip()
	return s


def process_data(n):
	with open('data/nyt_corpus_'+str(n)+'.csv') as f1:
		with open('data/nyt_corpus_cleaned_'+str(n)+'.csv', 'w') as f2:
			start = time.time()
			print('Creating new file')

			i = 0
			reader = csv.reader(f1)
			writer = csv.writer(f2)

			for line in reader:
				# Create cleaned line
				if i == 0:
					clean_line = ['desk', 'full_text', 'lead_paragraph', 'headline', 'nouns', 'lemmas', 'length']
				else:
					desk = clean_label(line[5])
					full_text = clean_input_column(line[6])
					headline = clean_input_column(line[8])
					lead_paragraph = clean_input_column(line[14])
					length = line[15]
					nouns, lemmas = create_nouns_lemmas(full_text)
					clean_line = [desk, full_text, lead_paragraph, headline, nouns, lemmas, length]
			
				# Write it to new file
				writer.writerow(clean_line)
				
				if i != 0 and i % 1000 == 0:
					elapsed = time.time() - start
					hours, rem = divmod(elapsed, 3600)
					minutes, seconds = divmod(rem, 60)
					print('Wrote {} cleaned records to CSV file {:0>2}:{:0>2}:{:0>2}'.format(str(i), int(hours), int(minutes), int(seconds)))
				
				i += 1

	print('Preprocessing complete')
	

if __name__ == '__main__':
	n = int(sys.argv[1])
	process_data(n)

