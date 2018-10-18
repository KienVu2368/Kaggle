from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import numpy as np
import pandas as pd
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.optimizers import *
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.optimizers import *
from keras.utils.vis_utils import plot_model

from utils import read_yaml, vocab_size




params = read_yaml()
toxic_types = params.toxic_types


def xgboost(xtrain, ytrain, xtest, params, num_rounds):
    preds = np.zeros((params.preds.test_len, params.preds.output_len))
    
    xgtest = xgb.DMatrix(xtest)
    for i, j in enumerate(toxic_types):
        print('fit ' + j + ' comment')
        model = _xgboost_model(xtrain, ytrain[j], params, num_rounds)
        print('predict ' + j + ' comment')
        preds[:,i] = model.predict(xgtest)
    
    return(preds)

def logistic(train_tfidf, train_data, test_tfidf, C = 10):
    model = LogisticRegression(C = C)
    preds = np.zeros((params.preds.test_len, params.preds.output_len))
    
    for i, j in enumerate(toxic_types):
        print('fit ' + j + ' comment')
        model = logistic() #LogisticRegression(C = 10)
        model.fit(train_tfidf, train_data[j])
        
        print('predict ' + j + ' comment')
        probs = model.predict_proba(test_tfidf)[:,1]
        preds[:,i] = probs 
    return(preds)

def essemble(predicted, geometric_mean = True):
    submit = pd.read_csv('submission/sample_submission.csv')
    if geometric_mean:
        print('geometric mean')
        submit[toxic_types] = 1
        for file_dir in predicted:
            result = pd.read_csv('submission/' + file_dir)
            submit[toxic_types] = submit[toxic_types]*result[toxic_types]
        submit[toxic_types]  = submit[toxic_types].apply(lambda x: x**(1/len(predicted)))
    else:
        print('arithmetical mean')
        submit[toxic_types] = 0
        for file_dir in predicted:
            result = pd.read_csv('submission/' + file_dir)
            submit[toxic_types] = submit[toxic_types] + result[toxic_types]
        submit[toxic_types]  = submit[toxic_types]/len(predicted)
    return(submit)


def blending(train_tfidf, valid_tfidf, y_train, y_valid, test_tfidf):
    #list of models
    models = []
    models.append([_logistic_model(C = C) for C in params.blending.models.Logistic.C])
    models.append([_MultinomialNB(alpha = alpha) for alpha in params.blending.models.MultinomialNB.alpha])
    
    preds = np.zeros((params.preds.test_len, params.preds.output_len))
    blend_x_valid = np.zeros((valid_tfidf.shape[0], len(models)*len(train_tfidf)))
    blend_x_submit = np.zeros((test_tfidf.shape[0], len(models)*len(train_tfidf)))
    
    #fit model
    for i, j in enumerate(toxic_types):
        print('predict ' + j + ' comment')
        m = -1
        for l in range(len(train_tfidf)):
            for k, model in enumerate(models):
                m += 1
                model.fit(train_tfidf[l], y_train[j])
                blend_x_valid[:,m] = model.predict_proba(valid_tfidf[l])[:,1]    
                blend_x_submit[:,m] = model.predict_proba(test_tfidf[l])[:,1] 
        
        blend_model = LogisticRegression()
        blend_model.fit(blend_x_valid, y_valid[j])
        
        
        probs = blend_model.predict_proba(blend_x_submit)[:,1]
        #probs = (probs - probs.min()) / (probs.max() - probs.min())
        preds[:,i] = probs
    
    return(preds)  
    

def CNN_wordlevel(x_train_pad, y_train, x_test_pad, vocab_size):
    model = _CNN_wordlevel_model(vocab_size,
                                 params.CNN_wordlevel.output_len,
                                 params.CNN_wordlevel.max_length, 
                                 params.CNN_wordlevel.max_range)
    
    model.fit(x_train_pad, 
              y_train, 
              params.CNN_wordlevel.epochs, 
              params.CNN_wordlevel.batch_size)
    
    preds = model.predict(x_test_pad)
    return(preds)
    
    
def CNN_subwordlevel(x_train_pad, y_train, x_test_pad, vocab_size):
    model = _CNN_wordlevel_model(vocab_size,
                                 params.CNN_subwordlevel.output_len,
                                 params.CNN_subwordlevel.max_length, 
                                 params.CNN_subwordlevel.max_range)
    
    model.fit(x_train_pad, 
              y_train, 
              params.CNN_subwordlevel.epochs, 
              params.CNN_subwordlevel.batch_size)
    
    preds = model.predict(x_test_pad)
    return(preds)
    
def _xgboost_model(xtrain, ytrain, params, num_rounds):
    #param
    plst = list(params.items())
    
    #features
    xgtrain = xgb.DMatrix(xtrain, label=ytrain)    
    
    #model
    model = xgb.train(plst, xgtrain, num_rounds)
    
    return(model)
    
    
def _logistic_model(C):
    return(LogisticRegression(C))

def _MultinomialNB(alpha):
    return(MultinomialNB(alpha))

def _CNN_wordlevel_model(vocab_size, output_len, max_length = 300, max_range = 7):
    #channel
    convs = []
    inputs = Input(shape=(max_length,))
    for kernel_size in range(1,max_range):        
        embedding = Embedding(vocab_size, 32)(inputs)
        conv = Conv1D(filters=32, kernel_size=kernel_size, activation='relu')(embedding)
        drop = SpatialDropout1D(0.2)(conv)
        pool = MaxPooling1D(pool_size=2)(drop)
        flat = Flatten()(pool)
        convs.append(flat)
    
    #merge
    merged = concatenate(convs)
    
    #interpretation
    dense = Dense(36, activation='relu')(merged)
    batchnorm = BatchNormalization()(dense)
    outputs = Dense(output_len, activation='sigmoid')(batchnorm)
    model = Model(inputs=inputs, outputs=outputs)
    
    #compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #print
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='model_preview/cnn_wordlevel.png')
    
    return(model)
    
    
def _CNN_subwordlevel_model(vocab_size, output_len, max_length = 300, max_range = 7):
    #channel
    convs = []
    inputs = Input(shape=(max_length,))
    for kernel_size in range(2,max_range):        
        embedding = Embedding(vocab_size, 32)(inputs)
        conv = Conv1D(filters=32, kernel_size=kernel_size, activation='relu')(embedding)
        drop = SpatialDropout1D(0.2)(conv)
        pool = MaxPooling1D(pool_size=2)(drop)
        flat = Flatten()(pool)
        convs.append(flat)
    
    #merge
    merged = concatenate(convs)
    
    #interpretation
    dense = Dense(36, activation='relu')(merged)
    batchnorm = BatchNormalization()(dense)
    outputs = Dense(output_len, activation='sigmoid')(batchnorm)
    model = Model(inputs=inputs, outputs=outputs)
    
    #compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #print
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='model_preview/cnn_subwordlevel.png')
    
    return(model)