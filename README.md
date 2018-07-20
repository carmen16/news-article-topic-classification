# W266 Final Project
### Carmen Easterwood

*Project Topic*: Topic classification for news articles

*Note*: The data consists of 1.8 million XML files and is **NOT** loaded into GitHub, as it far exceeds the file size limit.

1. `parse_corpus.py`: Selects a random subset of N articles, parses those articles, and saves the result as a CSV file (`data/nyt_corpus.csv`). User provides N at runtime, e.g. `python3 parse_corpus.py 1000`. When using Jupyter Notebook, N should probably not exceed 10,000, as it gets extremely slow to run the code.
2. `preprocessing.py`: Takes the file created in step 1, creates a nouns-only version of the articles and a lemmatized version of the articles, cleans the labels, then saves the relevant columns to a CSV file (`data/nyt_corpus_cleaned.csv`). The columns in the cleaned file are `full_text`, `lead_paragraph`, `nouns`, `lemmas`, and `desk`, where `desk` is the cleaned labels.
3. `split_vectorize.py`: Takes the file created in step 2, splits it into training/dev/test data, and TF-IDF vectorizes the articles. Results is `SplitVectorize` object which has attributes like `train_data`, `train_labels`, and `tv_train` (i.e. the TF-IDF vectorized version of the training data).
4. `base_models.py`: Takes a `SplitVectorize` object and runs it through a Multinomial NB or Logistic Regression model, testing different values of the parameters (`alpha`, `C`, `penalty`) and creating a table of model accuracies for different parameter values. Finally, graphs the accuracy of the model with different parameter values and different inputs (full text, lead paragraph, nouns, lemma) and prints the parameter/input combination with the highest accuracy.
