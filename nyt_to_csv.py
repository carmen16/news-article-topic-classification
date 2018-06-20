from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import csv



# Functions that look for individual data points for an article

def get_title(article_soup):
	try:
		title = article_soup.head.title.text
	except Exception:
		title = None
	finally:
		return title

def get_metadata(article_soup):
	metadata = {'publication_day_of_month': None,
		'publication_month': None,
		'publication_year': None,
		'publication_day_of_week': None,
		'dsk': None,
		'print_page_number': None,
		'print_section': None,
		'print_column': None,
		'online_sections': None}
	try:
		metas = article_soup.head.find_all('meta')
		for meta in metas:
			try:
				metadata[meta['name']] = int(meta['content'])
			except Exception:
				metadata[meta['name']] = meta['content']
	finally:
		return metadata

def get_id(article_soup):
	try:
		id_ = article_soup.head.docdata.find('doc-id')['id-string']
	except Exception:
		id_ = None
	finally:
		return id_

def get_copyright(article_soup, item):
	try:
		copyright_item = article_soup.head.docdata.find('doc.copyright')[item]
	except Exception:
		copyright_item = None
	finally:
		return copyright_item

def get_series(article_soup):
	try:
		series_name = article_soup.head.docdata.series['series.name']
	except Exception:
		series_name = None
	finally:
		return series_name

def get_length(article_soup):
	try:
		length = int(article_soup.head.pubdata['item-length'])
		length_unit = article_soup.head.pubdata['unit-of-measure']
	except Exception:
		length, length_unit = None, None
	finally:
		return length, length_unit

def get_indexing_service(article_soup, item_type):
	try:
		index_item = article_soup.head.docdata.find('identified-content').find(item_type, class_ = 'indexing_service').text
	except Exception:
		index_item = None
	finally:
		return index_item

def get_classifiers(article_soup):
	cl = {'indexing_service_descriptor': None,
		'online_producer_types_of_material': None,
		'online_producer_taxonomic_classifier': None,
		'online_producer_descriptor': None,
		'online_producer_general_descriptor': None}
	try:
		classifiers = article_soup.head.docdata.find('identified-content').find_all('classifier')
		for classifier in classifiers:
			if cl[classifier['class']+'_'+classifier['type']] == None:
				cl[classifier['class']+'_'+classifier['type']] = [classifier.text]
			else:
				cl[classifier['class']+'_'+classifier['type']].append(classifier.text)
	finally:
		return cl

def get_headline(article_soup):
	try:
		headline = article_soup.body.find('body.head').hedline.hl1.text
	except Exception:
		headline = None
	finally:
		return headline

def get_online_headline(article_soup):
	try:
		online_headline = article_soup.body.find('body.head').hedline.find('hl2', class_ = 'online_headline').text
	except Exception:
		online_headline = None
	finally:
		return online_headline

def get_byline(article_soup, byline_type):
	try:
		byline = article_soup.body.find('body.head').find('byline', class_ = byline_type).text
	except Exception:
		byline = None
	finally:
		return byline

def get_abstract(article_soup):
	try:
		strings = article_soup.body.find('body.head').abstract.stripped_strings
		abstract = ''
		for string in strings:
			abstract = abstract + ' ' + string
	except Exception:
		abstract = None
	finally:
		return abstract

def get_body_content(article_soup, content_type):
	try:
		strings = article_soup.body.find('body.content').find('block', class_ = content_type).stripped_strings
		s = ''
		for string in strings:
			s = s + ' ' + string
	except Exception:
		s = None
	finally:
		return s

def get_author_info(article_soup):
	try:
		author_info = article_soup.body.find('body.end').find('tagline', class_ = 'author_info').text
	except Exception:
		author_info = None
	finally:
		return author_info



# Gathers article data if available and saves to dictionary

def parse_article(article_soup, new_article_obj):

	# Get article attributes from soup
	metadata = get_metadata(article_soup)
	length, length_unit = get_length(article_soup)
	classifiers = get_classifiers(article_soup)

	new_article_obj['title'].append(get_title(article_soup))
	new_article_obj['publication_day_of_month'].append(metadata['publication_day_of_month'])
	new_article_obj['publication_month'].append(metadata['publication_month'])
	new_article_obj['publication_year'].append(metadata['publication_year'])
	new_article_obj['publication_day_of_week'].append(metadata['publication_day_of_week'])
	new_article_obj['desk'].append(metadata['dsk'])
	new_article_obj['print_page_number'].append(metadata['print_page_number'])
	new_article_obj['print_section'].append(metadata['print_section'])
	new_article_obj['print_column'].append(metadata['print_column'])
	new_article_obj['online_sections'].append(metadata['online_sections'])
	new_article_obj['id'].append(get_id(article_soup))
	new_article_obj['copyright_holder'].append(get_copyright(article_soup,'holder'))
	new_article_obj['copyright_year'].append(int(get_copyright(article_soup,'year')))
	new_article_obj['series_name'].append(get_series(article_soup))
	new_article_obj['indexing_descriptor'].append(classifiers['indexing_service_descriptor'])
	new_article_obj['indexing_location'].append(get_indexing_service(article_soup,'location'))
	new_article_obj['indexing_org'].append(get_indexing_service(article_soup,'org'))
	new_article_obj['indexing_person'].append(get_indexing_service(article_soup,'person'))
	new_article_obj['types_of_material'].append(classifiers['online_producer_types_of_material'])
	new_article_obj['taxonomic_classifier'].append(classifiers['online_producer_taxonomic_classifier'])
	new_article_obj['descriptor'].append(classifiers['online_producer_descriptor'])
	new_article_obj['general_descriptor'].append(classifiers['online_producer_general_descriptor'])
	new_article_obj['length'].append(length)
	new_article_obj['length_unit'].append(length_unit)
	new_article_obj['headline'].append(get_headline(article_soup))
	new_article_obj['online_headline'].append(get_online_headline(article_soup))
	new_article_obj['print_byline'].append(get_byline(article_soup,'print_byline'))
	new_article_obj['normalized_byline'].append(get_byline(article_soup,'normalized_byline'))
	new_article_obj['abstract'].append(get_abstract(article_soup))
	new_article_obj['lead_paragraph'].append(get_body_content(article_soup,'lead_paragraph'))
	new_article_obj['full_text'].append(get_body_content(article_soup,'full_text'))
	new_article_obj['author_info'].append(get_author_info(article_soup))



# Functions for choosing random articles

def create_file_index():
	df = pd.read_csv('nyt_corpus_docs/file.txt', names = ['filepath'])
	df_filtered = df.drop(df[~df['filepath'].str.endswith('.xml')].index)
	df_filtered.sort_values(by = ['filepath'], ascending=True, inplace=True)
	df_filtered.to_csv('file_index.txt', header=False, index=False)

def get_article_list(num_articles):
	article_df = pd.read_csv('file_index.txt', names = ['filepath'])
	article_list = list(article_df['filepath'].sample(num_articles))

	print('Got list of ' + str(num_articles) + ' random articles.')
	return article_list


# Converts a selected article to soup

def get_article(xml_text):
	soup = BeautifulSoup(xml_text, 'xml')
	return soup


# Saves final output to CSV

def save_data(article_obj, filename):
	df = pd.DataFrame(article_obj)
	df.to_csv(filename, index = False)


 
# Gets a list of random articles, parses them, and saves to a CSV file

def main():
	article_obj = {
		'title': []
		, 'publication_day_of_month': []
		, 'publication_month': []
		, 'publication_year': []
		, 'publication_day_of_week': []
		, 'desk': []
		, 'print_page_number': []
		, 'print_section': []
		, 'print_column': []
		, 'online_sections': []
		, 'id': []
		, 'copyright_holder': []
		, 'copyright_year': []
		, 'series_name': []
		, 'indexing_descriptor': []
		, 'indexing_location': []
		, 'indexing_org': []
		, 'indexing_person': []
		, 'types_of_material': []
		, 'taxonomic_classifier': []
		, 'descriptor': []
		, 'general_descriptor': []
		, 'length': []
		, 'length_unit': []
		, 'headline': []
		, 'online_headline': []
		, 'print_byline': []
		, 'normalized_byline': []
		, 'abstract': []
		, 'lead_paragraph': []
		, 'full_text': []
		, 'author_info': []
	}

	article_list = get_article_list(1000)

	for article in article_list:
		if article_list.index(article) % 100 == 0:
			print('Parsing article ' + str(article_list.index(article)))

		article_soup = get_article(open(article))
		parse_article(article_soup, article_obj)

	save_data(article_obj, 'nyt_corpus.csv')


if __name__ == '__main__':
	#create_file_index() # doesn't need to be run again
	main()



