import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences

from scipy import sparse
from scipy.sparse import coo_matrix, hstack

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from utils import read_yaml




params = read_yaml()
toxic_types = params.toxic_types


def tfidf(train_comment, test_comment, valid_comment = None, max_word_ngram = 2, max_char_ngram = 6, stack = True):
    #list of feature set
    tfidf_vectorizer = []
    for word_ngram in range(1, max_word_ngram +1):
        tfidf_vectorizer.append(TfidfVectorizer(use_idf=True, ngram_range=(word_ngram, word_ngram), sublinear_tf=True))
    for char_ngram in range(2, max_char_ngram + 1):
        tfidf_vectorizer.append(TfidfVectorizer(analyzer='char', ngram_range=(char_ngram, char_ngram), use_idf=True, sublinear_tf=True))
    
    train_tfidf_features = []
    test_tfidf_features = []
    valid_tfidf_features = []
    
    for vectorizer in tfidf_vectorizer:
        train_tfidf_features.append(vectorizer.fit_transform(train_comment))    
        test_tfidf_features.append(vectorizer.transform(test_comment))
        if valid_comment is not None:
            valid_tfidf_features.append(vectorizer.transform(valid_comment))
            
    if stack:
        train_tfidf_feature = sparse.hstack(train_tfidf_features)
        test_tfidf_feature = sparse.hstack(test_tfidf_features)    
        if valid_comment is not None:
            valid_tfidf_feature = sparse.hstack(valid_tfidf_features)
            return(train_tfidf_feature, valid_tfidf_feature, test_tfidf_feature)
        else:
            return(train_tfidf_feature, test_tfidf_feature)
    else:
        if valid_comment is not None:
            return(train_tfidf_features, valid_tfidf_features, test_tfidf_feature)
        else:
            return(train_tfidf_features, test_tfidf_feature)
    
def pad_sequence(train_comment, test_comment, vocab_size, max_length = 300):
    #tokenize
    tokenizer = text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(list(train_comment))
    
    # train data
    x_train_token = tokenizer.texts_to_sequences(train_comment)
    x_train_pad = sequence.pad_sequences(x_train_token, maxlen=max_length)
    
    #test data
    x_test_token = tokenizer.texts_to_sequences(test_comment)
    x_test_pad = sequence.pad_sequences(x_test_token, maxlen=max_length)
    
    return(x_train_pad, x_test_pad)
    
