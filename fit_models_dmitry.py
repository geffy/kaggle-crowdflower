# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utility import *
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import cfg



def build_model(titles,X1,X3,X4,titles_test,X1_test,X3_test,X4_test,y,weights=None,params=[400,10,0.0],top_words=10):
    '''
    X1: query lenght,title lenght,description presetn flag,number of words from query that also occured in title,
        compression distance between query and title ,1 - edit distance between query and title,
        1 - average(maximum edit distance between word from query and every word from title),
        last word from query present in title flag,ratio of words from query that also occured in title
    X3: Stanislav's features
    X4: Mikhail's features
    params list: [Number of SVD components, C in SVR, gamma in SVR]
    '''
    #get features from extended queries
    if top_words==10:
        X5 = np.loadtxt(cfg.path_features + 'train_ext_counts_top10.txt')
        X5_test = np.loadtxt(cfg.path_features + 'test_ext_counts_top10.txt')
        queries_ext = np.array(pd.read_csv(cfg.path_features + 'train_ext_top10.csv')['query'])
        queries_ext_test = np.array(pd.read_csv(cfg.path_features + 'test_ext_top10.csv')['query'])
    elif top_words==15:
        X5 = np.loadtxt(cfg.path_features + 'train_ext_counts_top15.txt')
        X5_test = np.loadtxt(cfg.path_features + 'test_ext_counts_top15.txt')
        queries_ext = np.array(pd.read_csv(cfg.path_features + 'train_ext_top15.csv')['query'])
        queries_ext_test = np.array(pd.read_csv(cfg.path_features + 'test_ext_top15.csv')['query'])
    else:
        print('Generate features for extended queries. top10 or top 15.')
        print(1/0)
    
    df_train = pd.DataFrame(np.c_[queries_ext,titles],columns=['query','product_title'])
    df_test = pd.DataFrame(np.c_[queries_ext_test,titles_test],columns=['query','product_title'])
    train_qt = list(df_train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    test_qt = list(df_test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    
    
    tfv = text.TfidfVectorizer(min_df=10,  max_features=None, 
            strip_accents='unicode', analyzer='char',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
            
    tfv.fit(train_qt)
    X2 =  tfv.transform(train_qt)
    X2_test = tfv.transform(test_qt)
    svd = TruncatedSVD(n_components=params[0])
    mms = MinMaxScaler()
    
    X = np.c_[svd.fit_transform(X2),X1,X4,X3,X5]
    X_test = np.c_[svd.transform(X2_test),X1_test,X4_test,X3_test,X5_test]
    
    X=mms.fit_transform(X)
    X_test = mms.transform(X_test)
    
    clf = SVR(C=params[1],gamma=params[2],cache_size=2048,kernel='rbf')
    clf.fit(X,y,sample_weight=weights)
    p = clf.predict(X_test)
    return p

train = pd.read_csv(cfg.path_train).fillna("")
test  = pd.read_csv(cfg.path_test ).fillna("")
idx = test.id.values.astype(int)
y = train.median_relevance.values

X1, weights, titles = (np.loadtxt(cfg.path_features + 'train_counts.txt'),
                       np.array(pd.read_csv(cfg.path_features + 'weights.csv'))[:,0],
                       np.array(pd.read_csv(cfg.path_features + 'titles_clean.csv'))[:,0])
X1_test, titles_test = (np.loadtxt(cfg.path_features + 'test_counts.txt'),
                        np.array(pd.read_csv(cfg.path_features + 'titles_test_clean.csv'))[:,0])


X4 = np.loadtxt(cfg.path_features + 'X_additional_tr.txt')
X4_test = np.loadtxt(cfg.path_features + 'X_additional_te.txt')

X3 = np.loadtxt(cfg.path_features + 'ssfeas4train.txt')
X3_test = np.loadtxt(cfg.path_features + 'ssfeas4test.txt')

np.random.seed(seed=22)
p1 = build_model(titles,X1,X3,X4,titles_test,X1_test,X3_test,X4_test,y,weights=weights,params=[300,8,0.15],top_words=10)

p2 = build_model(titles,X1,X3,X4,titles_test,X1_test,X3_test,X4_test,y,weights=weights,params=[400,4,0.20],top_words=15)

np.savetxt(cfg.path_features + 'dmitry_model1.txt',p1)
np.savetxt(cfg.path_features + 'dmitry_model2.txt',p2)
