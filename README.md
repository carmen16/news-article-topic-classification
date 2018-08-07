# Optimal Model Input for Newspaper Topic Classification
#### Carmen Easterwood
#### W266 Natural Language Processing with Deep Learning
#### Summer 2018

## Final Report

+ See: [reports/final_report/final-report.pdf](https://github.com/carmen16/w266-final-project/blob/master/reports/final_report/final-report.pdf)
+ The project proposal and project milestone are also in [reports/](https://github.com/carmen16/w266-final-project/tree/master/reports)
+ Everything referenced in the final report is in [references/](https://github.com/carmen16/w266-final-project/tree/master/references)

## Code Summary

Below is a quick summary of each python file. See the Jupyter notebook [Code_Walkthrough.ipynb](https://github.com/carmen16/w266-final-project/blob/master/Code_Walkthrough.ipynb) to walk through the code and its results. 

1. [parse_corpus.py](https://github.com/carmen16/w266-final-project/blob/master/parse_corpus.py): Selects a random subset of N articles, parses those articles, and saves the result as a CSV file (`data/nyt_corpus_N.csv`). The user provides N at runtime, e.g. `python3 parse_corpus.py 1000`. The file is saved in the [data/](https://github.com/carmen16/w266-final-project/tree/master/data) folder. I have created files with N = 1,000 and N = 10,000 and pushed them to [data/](https://github.com/carmen16/w266-final-project/tree/master/data). I also work with larger N in [Code_Walkthrough.ipynb](https://github.com/carmen16/w266-final-project/blob/master/Code_Walkthrough.ipynb), but these files are too big to push to GitHub.
2. [preprocessing.py](https://github.com/carmen16/w266-final-project/blob/master/preprocessing.py): Takes a file created in step 1, creates a nouns-only version of the news articles and a lemmatized version of the news articles, cleans up the labels, then saves the relevant columns to a CSV file (`data/nyt_corpus_cleaned_N.csv`). The user needs to provide N at runtime so the program knows which file from step 1 to clean (`python3 preprocessing.py 1000`). The columns in the cleaned file are:
    + 1 model output: `desk`
    + 5 model inputs: `full_text`, `lead_paragraph`, `headlines`, `nouns`, `lemmas`
3. [split_vectorize.py](https://github.com/carmen16/w266-final-project/blob/master/split_vectorize.py): Takes one of the files created in step 2, splits it into training/dev/test data, and TF-IDF vectorizes the news articles. The result is a `SplitVectorize` object which has attributes like `train_data`, `train_labels`, and `tv_train` (i.e. the TF-IDF vectorized version of the training data). More attributes will be added to the object in later steps.
    + The code is set up this way so that each model input can have its own object that accumulates different models and statistics
4. [base_models.py](https://github.com/carmen16/w266-final-project/blob/master/base_models.py): Takes a `SplitVectorize` object from step 3 and runs it through a Multinomial NB or Logistic Regression model, testing different values of the parameters (`alpha`, `C`, `penalty`) and creating a table of model accuracies for different parameter values. Also allows the user to graph the accuracy of the model with different parameter values and different model inputs.
    + Plots are saved to the [plots/](https://github.com/carmen16/w266-final-project/tree/master/plots) folder
5. [neural_net.py](https://github.com/carmen16/w266-final-project/blob/master/neural_net.py): Loads GloVe word embeddings from [GloVe/](https://github.com/carmen16/w266-final-project/tree/master/GloVe) to initialize the embedding matrix for the neural net. Then takes a `SplitVectorize` object from step 3, tokenizes the model input, pads the tokens, and runs the padded tokens through a Convolutional Neural Network model.

## Where is the Data?

+ The data consists of 1.8 million XML files and is **NOT** loaded into GitHub, as it far exceeds the file size limit.
    + You can download the data at: the [Linguistic Data Consortium](https://catalog.ldc.upenn.edu/ldc2008t19) or from the UC Berkeley library
    + My code refers to the file path `data/ldc/*`, so the data needs to be placed there for `parse_corpus.py` (step 1) to work.
+ I have uploaded some CSV files with random samples of the data ([data/nyt_corpus_1000.csv](https://github.com/carmen16/w266-final-project/blob/master/data/nyt_corpus_1000.csv) and [data/nyt_corpus_10000.csv](https://github.com/carmen16/w266-final-project/blob/master/data/nyt_corpus_10000.csv)). See Step 1 in [Code Summary](#code-summary) for more info.
