# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import backports.lzma as lzma
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text
from difflib import SequenceMatcher as seq_matcher
from itertools import combinations_with_replacement
from sklearn.preprocessing import MinMaxScaler
import re
from collections import Counter

def construct_extended_query(queries,queries_test,titles,titles_test,top_words=10):
    y = pd.read_csv('raw/train.csv').median_relevance.values
    
    stop_words = text.ENGLISH_STOP_WORDS
    pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
    
    train = pd.read_csv('raw/train.csv')
    test = pd.read_csv('raw/test.csv')
    
    data = []
    query_ext_train = np.zeros(len(train)).astype(np.object)
    query_ext_test = np.zeros(len(test)).astype(np.object)
    for q in np.unique(queries):
        q_mask = queries == q
        q_test = queries_test == q
        
        titles_q = titles[q_mask]
        y_q = y[q_mask]
        
        good_mask = y_q > 3
        titles_good = titles_q[good_mask]
        ext_q = str(q)
        for item in titles_good:
            ext_q += ' '+str(item)
        ext_q = pattern.sub('', ext_q)
        c = [word for word, it in Counter(ext_q.split()).most_common(top_words)]
        c = ' '.join(c)
        data.append([q,ext_q,c])
        query_ext_train[q_mask] = c
        query_ext_test[q_test] = c
    
    train['query'] = query_ext_train
    test['query'] = query_ext_test
    train['product_title'] = titles
    test['product_title'] = titles_test
    return train, test
    
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)
    
def compression_distance(x,y,l_x=None,l_y=None):
    if x==y:
        return 0
    x_b = x.encode('utf-8')
    y_b = y.encode('utf-8')
    if l_x is None:
        l_x = len(lzma.compress(x_b))
        l_y = len(lzma.compress(y_b))
    l_xy = len(lzma.compress(x_b+y_b))
    l_yx = len(lzma.compress(y_b+x_b))
    dist = (min(l_xy,l_yx)-min(l_x,l_y))/max(l_x,l_y)
    return dist

def get_scores(std_true, y_true):
    best_diff = np.inf
    combs = list(combinations_with_replacement([1,2,3,4],3)) + list(combinations_with_replacement([1,2,3,4],4)) + list(combinations_with_replacement([1,2,3,4],5))
    for item in combs:
        if np.median(item) == y_true:
            diff = np.abs(np.std(item) - std_true)
            if diff < best_diff:
                best_diff = diff
                best_match = list(item)
                if best_diff < 1e-8:
                    break
    return best_match

def extend_set(X,y,weights):
    X_tr = []
    y_tr = []
    y_true = []
    for i in range(len(y)):
        std = 1/weights[i] - 1
        best_match = get_scores(std,y[i])
        y_true_vals = []
        for item in best_match:
            X_tr.append(X[i])
            y_tr.append(item)
            y_true_vals.append(False)
        for j in range(len(best_match)):
            if best_match[j]==y[i]:
                y_true_vals[j] = True
                break
        y_true += y_true_vals
    y_true = np.array(y_true)
    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)
    return X_tr, y_tr


def correct_string(s):
    s = s.replace("hardisk", "hard drive")
    s = s.replace("extenal", "external")
    s = s.replace("soda stream", "sodastream")
    s = s.replace("fragance", "fragrance")
    s = s.replace("16 gb", "16gb")
    s = s.replace("32 gb", "32gb")
    s = s.replace("500 gb", "500gb")
    s = s.replace("2 tb", "2tb")
    s = s.replace("shoppe", "shop")
    s = s.replace("refrigirator", "refrigerator")
    s = s.replace("assassinss", "assassins")
    s = s.replace("harleydavidson", "harley davidson")
    s = s.replace("harley-davidson", "harley davidson")
    return s
    
stemmer = PorterStemmer()

## Stemming functionality
class stemmerUtility(object):
    #Stemming functionality
    @staticmethod
    def stemPorter(review_text):
        porter = PorterStemmer()
        preprocessed_docs = []
        for doc in review_text:
            final_doc = []
            for word in doc:
                final_doc.append(porter.stem(word))
                #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
            preprocessed_docs.append(final_doc)
        return preprocessed_docs

def assemble_counts(train,m='train'):
    X = []
    titles = []
    queries = []
    weights = []
    train['isdesc'] = 1 # Description present flag
    train.loc[train['product_description'].isnull(),'isdesc'] = 0
    
    for i in range(len(train.id)):
        query = correct_string(train['query'][i].lower())
        title = correct_string(train.product_title[i].lower())
        
        query = (" ").join([z for z in BeautifulSoup(query).get_text(" ").split(" ")])
        title = (" ").join([z for z in BeautifulSoup(title).get_text(" ").split(" ")])
        
        query=text.re.sub("[^a-zA-Z0-9]"," ", query)
        title=text.re.sub("[^a-zA-Z0-9]"," ", title)
        
        query= (" ").join([stemmer.stem(z) for z in query.split(" ")])
        title= (" ").join([stemmer.stem(z) for z in title.split(" ")])

        query=" ".join(query.split())
        title=" ".join(title.split())
        
        dist_qt = compression_distance(query,title)
        dist_qt2 = 1 - seq_matcher(None,query,title).ratio()
        
        query_len = len(query.split())
        title_len = len(title.split())
        isdesc = train.isdesc[i]
        
        tmp_title = title
        word_counter_qt = 0
        lev_dist_arr = []
        for q in query.split():
            lev_dist_q = []
            for t in title.split():
                lev_dist = seq_matcher(None,q,t).ratio()
                if lev_dist > 0.9:
                    word_counter_qt += 1
                    #tmp_title += ' '+q # add such words to title to increase their weights in tfidf
                lev_dist_q.append(lev_dist)
            lev_dist_arr.append(lev_dist_q)
        last_word_in = 0
        for t in title.split():
            lev_dist = seq_matcher(None,query.split()[-1],t).ratio()
            if lev_dist > 0.9: 
                last_word_in = 1
        lev_max = 0
        for item in lev_dist_arr:
            lev_max_q = max(item)
            lev_max += lev_max_q
        lev_max = 1- lev_max/len(lev_dist_arr)
        word_counter_qt_norm = word_counter_qt/query_len
        
        
        
        X.append([query_len,title_len,isdesc,word_counter_qt,dist_qt,dist_qt2,lev_max,last_word_in,word_counter_qt_norm])
        titles.append(tmp_title)
        queries.append(query)
        if m =='train':
            weights.append(1/(float(train["relevance_variance"][i]) + 1.0))
    X = np.array(X).astype(np.float)
    if m =='train':
        return X, np.array(weights).astype(np.float), np.array(titles), np.array(queries)
    else:
        return X, np.array(titles), np.array(queries)
        
def assemble_counts2(train):
    X = []
    queries = []
    
    for i in range(len(train.id)):
        query = train['query'][i]
        title = train.product_title[i]
        
        dist_qt = compression_distance(query,title)
        dist_qt2 = 1 - seq_matcher(None,query,title).ratio()
        
        query_len = len(query.split())
        
        lev_dist_arr = []
        word_rank_list = []
        word_q_ind = 0
        word_counter_qt = 0
        for q in query.split():
            word_q_ind += 1
            lev_dist_q = []
            for t in title.split():
                lev_dist = seq_matcher(None,q,t).ratio()
                if lev_dist > 0.9:
                    word_counter_qt += 1
                    word_rank_list.append(word_q_ind)
                    #tmp_title += ' '+q # add such words to title to increase their weights in tfidf
                lev_dist_q.append(lev_dist)
            lev_dist_arr.append(lev_dist_q)
        if word_counter_qt == 0:
            maxrank = 0
        else:
            maxrank = 26 - min(word_rank_list)
        
        
        lev_max = 0
        for item in lev_dist_arr:
            lev_max_q = max(item)
            lev_max += lev_max_q
        lev_max = 1- lev_max/len(lev_dist_arr)
        word_counter_qt_norm = word_counter_qt/query_len
        
        
        
        X.append([word_counter_qt,dist_qt,dist_qt2,lev_max,word_counter_qt_norm,maxrank])
        queries.append(query)

    X = np.array(X).astype(np.float)

    return X, np.array(queries)
    

def vary_border(pred_true,y,num_iter=101):
    mms = MinMaxScaler()
    pred=pred_true.copy()
    pred=mms.fit_transform(pred)
    best_score = 0
    for k1 in range(num_iter):
        c1 = k1/(num_iter-1)
        for k2 in range(num_iter):
            c2 = k2/(num_iter-1)
            for k3 in range(num_iter):
                c3 = k3/(num_iter-1)
                if c1 < c2 and c1 < c3 and c2 < c3 and c1 > 0.25 and c1 < 0.5 and c3 < 0.9:
                    tmp_pred = pred.copy()
                    mask1 = tmp_pred < c1
                    mask2 = (tmp_pred >=c1) * (tmp_pred < c2)
                    mask3 = (tmp_pred >=c2) * (tmp_pred < c3)
                    mask4 = tmp_pred >=c3
                    tmp_pred[mask1] = 1
                    tmp_pred[mask2] = 2
                    tmp_pred[mask3] = 3
                    tmp_pred[mask4] = 4
                    score = quadratic_weighted_kappa(y,tmp_pred)
                    if score > best_score:
                        best_score = score
                        best_coef = [c1,c2,c3]
                        best_pred = tmp_pred.copy()
    #print(best_score,best_coef)
    return best_pred, best_coef

def apply_border(pred,coefs):
    c1, c2, c3 = coefs[0], coefs[1], coefs[2]
    mms2 = MinMaxScaler()
    tmp_pred=mms2.fit_transform(pred)
    mask1 = tmp_pred < c1
    mask2 = (tmp_pred >=c1) * (tmp_pred < c2)
    mask3 = (tmp_pred >=c2) * (tmp_pred < c3)
    mask4 = tmp_pred >=c3
    tmp_pred[mask1] = 1
    tmp_pred[mask2] = 2
    tmp_pred[mask3] = 3
    tmp_pred[mask4] = 4
    return tmp_pred.astype(np.int32)
    
