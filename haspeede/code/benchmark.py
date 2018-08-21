# -*- coding: utf-8 -*-
"""

@author: dadangewp
"""

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import logging
import codecs
import numpy as np

from mlp import MLP
from mlp_tf import create_model, train


logging.basicConfig(level=logging.INFO)


def parse_training(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = line.split("\t")[2]
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def featurize(corpus):
    '''
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''
    
    tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                              analyzer = "word",
                                          stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
    feature  = []
    for tweet in corpus:
        feature.append(tweet)
    
    tfidfVectorizer = tfidfVectorizer.fit(feature)
    X_train = tfidfVectorizer.transform(feature)
    return X_train

def train_mlp_pytorch(X_train, y_train, X_test, y_test):
    model = MLP([X_train.shape[1], 30, 10, 2])
    model.train(X_train, y_train, X_test, y_test, num_epochs=10000)

def train_mlp_tf(X_train, y_train, X_test, y_test):
    layer_sizes = [X_train.shape[1], 500, 100, 30, 10, 2]
    lr = 1e-5
    batch_size = 32
    run_id = 'layers_'str(layer_sizes) + '_lr_' + str(lr) + '_batch_' + batch_size
    model = create_model([X_train.shape[1], 30, 10, 2])
    train(model, X_train, y_train, X_test, y_test, run_id=run_id)


if __name__ == "__main__":
    # Experiment settings

    # Dataset: AMI-IberEval 2018
    #DIR_TRAIN = "D:\\PhD\\Hate Speech Evalita\\haspeede2018-master\\haspeede_FB-train.tsv"
    DIR_TRAIN = "../data/haspeede_FB-train.tsv"

    K_FOLDS = 10 # 10-fold crossvalidation
    clf = LinearSVC() # the default, non-parameter optimized linear-kernel SVM


    # Loading dataset and featurised simple Tfidf-BoW model
    corpus, y = parse_training(DIR_TRAIN)
    
    # Getting BoW feature
    X = featurize(corpus)
    
    # Training Classifier
    #clf.fit(X,y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    train_mlp_tf(X_train, y_train, X_test, y_test)
            
    # Cross Validation
    #scores = cross_val_score(clf, X, y, cv=K_FOLDS, scoring="accuracy")
    #print(scores)
    #print("Accuracy Score (Cross-V): %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    
