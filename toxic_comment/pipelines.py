import numpy as np
import pandas as pd
from utils import load_data, save_data, read_yaml, vocab_size
from features import tfidf, pad_sequence
from manipulation import clean_word, clean_subword, blending_data_split
from models import logistic, xgboost, essemble, blending, CNN_wordlevel, CNN_subwordlevel
import xgboost as xgb


#params
params = read_yaml()
toxic_types = params.toxic_types

#read data
train_data, test_data = load_data()

#logistic
def run_logistic():
    print('data manipulation')
    train_comment = train_data['comment_text'].apply(clean_word)
    test_comment = test_data['comment_text'].apply(clean_word)
    
    print('create features')
    train_tfidf, test_tfidf = tfidf(train_comment, test_comment)
    
    print('run logistic model')
    preds = logistic(train_tfidf, 
                     train_data, 
                     test_tfidf, 
                     params.logistic.C)
    print('save data')
    save_data(file_name = 'logistic_1', 
              preds = preds, 
              toxic_types = toxic_types)

#xgboost
def run_xgboost():
    print('data manipulation')
    train_comment = train_data['comment_text'].apply(clean_word)
    test_comment = test_data['comment_text'].apply(clean_word)
    
    print('create features')
    train_tfidf, test_tfidf = tfidf(train_comment, test_comment)
    
    print('run xgboost model')    
    preds = xgboost(train_tfidf, 
                    train_data, 
                    test_tfidf, 
                    params.xgboost.param, 
                    params.xgboost.num_rounds)    
    print('save data')
    save_data(file_name = 'xgboost_1', 
              preds = preds, 
              toxic_types = toxic_types)

#essemble
def run_essemble(predicted, geometric_mean = True):    
    print('run essemble')
    submit = essemble(predicted)  
    print('save data')
    submit.to_csv('submission/essemble_1.csv', index=False)
        
    
def run_blending():
    print('data manipulation')
    train_comment = train_data['comment_text'].apply(clean_word)
    test_comment = test_data['comment_text'].apply(clean_word)
    
    print('split data')
    x_train, x_valid, y_train, y_valid = blending_data_split(train_comment, 
                                                            train_data[toxic_types], 
                                                            params.blending.data_split.test_size,
                                                            params.blending.data_split.ramdom_state)
    print('create features')
    train_tfidf, valid_tfidf, test_tfidf = tfidf(x_train, 
                                                 test_comment,
                                                 x_valid, 
                                                 params.blending.tfidf.max_word_ngram,
                                                 params.blending.tfidf.max_char_ngram,
                                                 params.blending.tfidf.stack)
    print('run blending')
    preds = blending(train_tfidf, valid_tfidf, y_train, y_valid, test_tfidf)
    
    print('save data')
    save_data(file_name = 'bleding_1', 
              preds = preds, 
              toxic_types = toxic_types)
    
def run_CNN_wordlevel():
    print('data manipulation')
    train_comment = train_data['comment_text'].apply(clean_word)
    test_comment = test_data['comment_text'].apply(clean_word)
    vocab_size = vocab_size(train_comment)
    
    print('pad sequence')
    x_train_pad, x_test_pad = pad_sequence(train_comment, 
                                           test_comment, 
                                           vocab_size, 
                                           max_length = params.CNN_wordlevel.max_length)
    print('run CNN word level')
    preds = CNN_wordlevel(x_train_pad,
                          train_data[toxic_types],
                          x_test_pad,
                          vocab_size)
    
    print('save data')
    save_data(file_name = 'CNN_wordlevel_1', 
              preds = preds, 
              toxic_types = toxic_types)

def run_CNN_subwordlevel():
    print('data manipulation')
    train_comment = train_data['comment_text'].apply(clean_subword)
    test_comment = test_data['comment_text'].apply(clean_subword)
    vocab_size = vocab_size(train_comment)
    
    print('pad sequence')
    x_train_pad, x_test_pad = pad_sequence(train_comment, 
                                           test_comment, 
                                           vocab_size, 
                                           max_length = params.CNN_wordlevel.max_length)
    print('run CNN word level')
    preds = CNN_subwordlevel(x_train_pad,
                             train_data[toxic_types],
                             x_test_pad,
                             vocab_size)
    
    print('save data')
    save_data(file_name = 'CNN_subwordlevel_1',
              preds = preds, 
              toxic_types = toxic_types)
