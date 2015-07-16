import cfg
import kappa as pykappa

import pandas as pd
import numpy as np
import scipy.sparse as sp
import cPickle as pickle

import re
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from difflib import SequenceMatcher as seq_matcher

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from sklearn.cross_validation import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, LinearSVR

import logging
logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
logging.info("Mikhail's model_2 learner script")

# load data from dump
logging.info("load data")
train_df = pd.read_pickle(cfg.path_processed + 'train_df')
test_df =  pd.read_pickle(cfg.path_processed + 'test_df')
sampleSubmission = pd.read_csv(cfg.path_sampleSubmission)
# fetch Dmitry's extended querys
ext_train_df = pd.read_csv(cfg.path_features + 'train_ext_top10.csv')
ext_test_df =  pd.read_csv(cfg.path_features + 'test_ext_top10.csv')
train_df['title_ext'] = ext_train_df['product_title']
test_df['title_ext'] = ext_test_df['product_title']

logging.info("Collect BoWs")
# collect BoWs -- for query
vect = TfidfVectorizer(ngram_range=(1,2), min_df=2, encoding='utf-8')
vect.fit(list(train_df['query_stem'].values) + list(test_df['query_stem'].values)) 
X_query_tr = vect.transform(train_df['query_stem'].values)
X_query_te = vect.transform(test_df['query_stem'].values)
# fot title
vect.fit(list(train_df['title_stem'].values) + list(test_df['title_stem'].values))
X_tmp_tr = vect.transform(train_df['title_stem'].values).tocsc()
X_tmp_te = vect.transform(test_df['title_stem'].values).tocsc()
freq_tr = np.array(X_tmp_tr.sum(axis=0))[0]
freq_te = np.array(X_tmp_te.sum(axis=0))[0]
col_mask = np.where((freq_tr * freq_te)!=0)[0]
X_title_tr = X_tmp_tr[:, col_mask].tocsr()
X_title_te = X_tmp_te[:, col_mask].tocsr()
# for description
vect.fit(list(train_df['desc_stem'].values) + list(test_df['desc_stem'].values))
X_tmp_tr = vect.transform(train_df['desc_stem'].values).tocsc()
X_tmp_te = vect.transform(test_df['desc_stem'].values).tocsc()
freq_tr = np.array(X_tmp_tr.sum(axis=0))[0]
freq_te = np.array(X_tmp_te.sum(axis=0))[0]
col_mask = np.where((freq_tr * freq_te)!=0)[0]
X_desc_tr = X_tmp_tr[:, col_mask].tocsr()
X_desc_te = X_tmp_te[:, col_mask].tocsr()
# assemble in one
X_all_tr = sp.hstack([X_query_tr, X_title_tr, X_desc_tr]).tocsr()
X_all_te = sp.hstack([X_query_te, X_title_te, X_desc_te]).tocsr()

# coding query by id
le = LabelEncoder()
le.fit(train_df['query_stem'].values)
qid_tr = le.transform(train_df['query_stem'].values)
qid_te = le.transform(test_df['query_stem'].values)
y_all_tr = train_df['median_relevance'].values


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
            preprocessed_docs.append(final_doc)
        return preprocessed_docs

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
    

def assemble_counts(train):
    X = []
    titles = []
    for i in range(len(train.id)):
        query = correct_string(train['query'][i].lower())
        title = correct_string(train.product_title[i].lower())
        
        query = (" ").join([z for z in BeautifulSoup(query).get_text(" ").split(" ")])
        title = (" ").join([z for z in BeautifulSoup(title).get_text(" ").split(" ")])
        
        query=re.sub("[^a-zA-Z0-9]"," ", query)
        title=re.sub("[^a-zA-Z0-9]"," ", title)
        
        query= (" ").join([stemmer.stem(z) for z in query.split(" ")])
        title= (" ").join([stemmer.stem(z) for z in title.split(" ")])

        query=" ".join(query.split())
        title=" ".join(title.split())
        
        #dist_qt = compression_distance(query,title)
        dist_qt2 = 1 - seq_matcher(None,query,title).ratio()
        
        query_len = len(query.split())
        title_len = len(title.split())
        
        tmp_title = title
        word_counter_qt = 0
        lev_dist_arr = []
        for q in query.split():
            lev_dist_q = []
            for t in title.split():
                lev_dist = seq_matcher(None,q,t).ratio()
                if lev_dist > 0.9:
                    word_counter_qt += 1
                    tmp_title += ' '+q # add such words to title to increase their weights in tfidf
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
        X.append([query_len,title_len,word_counter_qt,lev_max,last_word_in,word_counter_qt_norm, dist_qt2])
        titles.append(tmp_title)
        
    X = np.array(X).astype(np.float)
    return X, np.array(titles)

logging.info("Assemble counts")
X_counts_tr, titles_tr = assemble_counts(train_df)
X_counts_te, titles_te = assemble_counts(test_df)

logging.info("Assemble additional features")
feat_list = [
    u'w2v_sim',
    u'w2v_dist',
    u'tsne_title_1', 
    u'tsne_title_2', 
    u'tsne_qt_1',
    u'tsne_qt_2',
    u'cosine_qt_orig', 
    u'cosine_qt_stem', 
    u'cosine_qd_stem',
    u'set_qt_stem'
]
X_additional_tr = train_df[feat_list].as_matrix()
X_additional_te = test_df[feat_list].as_matrix()

logging.info("Load w2v-based features")
X_w2v_tr, X_w2v_te = pickle.load(open(cfg.path_processed + 'X_w2v.pickled'))

logging.info("Load counts")
X_counts2_tr = np.loadtxt(cfg.path_features + 'train_ext_counts_top10.txt')
X_counts2_te = np.loadtxt(cfg.path_features + 'test_ext_counts_top10.txt')


# Learinig routines (similar to model1 -- see description here)
feat_list = [u'w2v_sim',            
             u'cosine_qt_stem',      
             u'cosine_qd_stem',         
             u'set_qt_stem',
             u'tsne_title_1',        
             u'tsne_title_2',           
             u'tsne_qt_1',
             u'tsne_qt_2',
            ]

def make_mf_sliced_regression(subset_tr, subset_te, clf, n_round=3, target_col='median_relevance'):
    print '\n [make_mf_slice]'
    print clf
    mf_tr = np.zeros(len(subset_tr))
    mf_te = np.zeros(len(subset_te))
    #query-slice
    for cur_query in subset_tr.query_stem.value_counts().index:
        mask_tr = subset_tr.query_stem == cur_query
        mask_te = subset_te.query_stem == cur_query
        
        # build Bow
        vect = CountVectorizer(min_df=1, ngram_range=(1,2))

        txts = (list((subset_tr[mask_tr]['title_ext']).values) + 
                list((subset_te[mask_te]['title_ext']).values))
        vect.fit(txts)

        X_loc_base = vect.transform(list((subset_tr[mask_tr]['title_ext']).values)).todense()
        X_loc_hold = vect.transform(list((subset_te[mask_te]['title_ext']).values)).todense()
        y_loc_train = subset_tr[mask_tr][target_col].values
        # intersect terms
        feat_counts = np.array(np.sum(X_loc_base, axis=0))[0] * np.array(np.sum(X_loc_hold, axis=0))[0]
        feat_mask = np.where(feat_counts>0)[0]
        # build final feats matrix
        X_loc_base = np.hstack((X_loc_base[:, feat_mask], subset_tr[mask_tr][feat_list]))
        X_loc_hold = np.hstack((X_loc_hold[:, feat_mask], subset_te[mask_te][feat_list]))
        
        # metafeatures iterators
        tmp_tr = np.zeros(sum(mask_tr))
        tmp_te = np.zeros(sum(mask_te))
        
        #print y_loc_train.shape, X_loc_base.shape
        
        for i in range(n_round):
            kf = KFold(len(y_loc_train), n_folds=2, shuffle=True, random_state=42+i*1000)
            for ind_tr, ind_te in kf:
                X_tr = X_loc_base[ind_tr]
                X_te = X_loc_base[ind_te]
                y_tr = y_loc_train[ind_tr]
                y_te = y_loc_train[ind_te]

                clf.fit(X_tr, y_tr)
                tmp_tr[ind_te] += clf.predict(X_te)
                tmp_te += clf.predict(X_loc_hold)*0.5
        mf_tr[mask_tr.values] = tmp_tr / n_round
        mf_te[mask_te.values] = tmp_te / n_round

    y_valid = subset_tr[target_col].values
    kappa = pykappa.quadratic_weighted_kappa(y_valid, np.round(mf_tr))
    acc = np.mean(y_valid == np.round(mf_tr))
    print '[{}] kappa:{}, acc:{}'.format(i, kappa, acc)
    return (mf_tr, mf_te)


def make_mf_sliced_classification(subset_tr, subset_te, clf, n_round=3, target_col='median_relevance'):
    print '\n [make_mf_slice]'
    print clf
    mf_tr = np.zeros(len(subset_tr))
    mf_te = np.zeros(len(subset_te))

    #query-slice
    for cur_query in subset_tr.query_stem.value_counts().index:
        mask_tr = subset_tr.query_stem == cur_query
        mask_te = subset_te.query_stem == cur_query
        
        # build Bow
        vect = CountVectorizer(min_df=1, ngram_range=(1,2))

        txts = (list((subset_tr[mask_tr]['title_ext']).values) + 
                list((subset_te[mask_te]['title_ext']).values))
        vect.fit(txts)

        X_loc_base = vect.transform(list((subset_tr[mask_tr]['title_ext']).values)).todense()
        X_loc_hold = vect.transform(list((subset_te[mask_te]['title_ext']).values)).todense()
        y_loc_train = subset_tr[mask_tr][target_col].values
        # intersect terms
        feat_counts = np.array(np.sum(X_loc_base, axis=0))[0] * np.array(np.sum(X_loc_hold, axis=0))[0]
        feat_mask = np.where(feat_counts>0)[0]
        # build final feats matrix
        X_loc_base = np.hstack((X_loc_base[:, feat_mask], subset_tr[mask_tr][feat_list]))
        X_loc_hold = np.hstack((X_loc_hold[:, feat_mask], subset_te[mask_te][feat_list]))
        
        # metafeatures iterators
        tmp_tr = np.zeros(sum(mask_tr))
        tmp_te = np.zeros(sum(mask_te))
        
        #print y_loc_train.shape, X_loc_base.shape
        
        for i in range(n_round):
            kf = KFold(len(y_loc_train), n_folds=2, shuffle=True, random_state=42+i*1000)
            for ind_tr, ind_te in kf:
                X_tr = X_loc_base[ind_tr]
                X_te = X_loc_base[ind_te]
                y_tr = y_loc_train[ind_tr]
                y_te = y_loc_train[ind_te]

                clf.fit(X_tr, y_tr)
                tmp_tr[ind_te] += clf.predict(X_te)
                tmp_te += clf.predict(X_loc_hold)*0.5
        mf_tr[mask_tr.values] = tmp_tr / n_round
        mf_te[mask_te.values] = tmp_te / n_round

    y_valid = subset_tr[target_col].values
    kappa = pykappa.quadratic_weighted_kappa(y_valid, np.round(mf_tr))
    acc = np.mean(y_valid == np.round(mf_tr))
    print '[{}] kappa:{}, acc:{}'.format(i, kappa, acc)
    return (mf_tr, mf_te)


def make_mf_regression(X ,y, clf, qid, X_test, n_round=3):
    print clf
    mf_tr = np.zeros(X.shape[0])
    mf_te = np.zeros(X_test.shape[0])
    for i in range(n_round):
        skf = StratifiedKFold(qid, n_folds=2, shuffle=True, random_state=42+i*1000)
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]

            y_tr = y[ind_tr]
            y_te = y[ind_te]

            clf.fit(X_tr, y_tr)
            mf_tr[ind_te] += clf.predict(X_te)
            mf_te += clf.predict(X_test)*0.5

            y_pred = np.round(clf.predict(X_te))
            kappa = pykappa.quadratic_weighted_kappa(y_te, y_pred)
            acc = np.mean(y_te == y_pred)
            print 'pred[{}] kappa:{}, acc:{}'.format(i, kappa, acc)
    return (mf_tr / n_round, mf_te / n_round)


def make_mf_classification4(X ,y, clf, qid, X_test, n_round=3):
    print clf
    mf_tr = np.zeros((X.shape[0], 5))
    mf_te = np.zeros((X_test.shape[0], 5))
    for i in range(n_round):
        skf = StratifiedKFold(qid, n_folds=2, shuffle=True, random_state=42+i*1000)
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]

            y_tr = y[ind_tr]
            y_te = y[ind_te]

            clf.fit(X_tr, y_tr)
            mf_tr[ind_te, 4] += clf.predict(X_te)
            mf_te[:, 4] += clf.predict(X_test)*0.5
            try:
                mf_tr[ind_te, :4] += clf.predict_proba(X_te)
                mf_te[:, :4] += clf.predict_proba(X_test)*0.5
            except:
                mf_tr[ind_te, :4] += clf.decision_function(X_te)
                mf_te[:,:4] += clf.decision_function(X_test)*0.5
            y_pred = np.round(clf.predict(X_te))
            kappa = pykappa.quadratic_weighted_kappa(y_te, y_pred)
            acc = np.mean(y_te == y_pred)
            print 'prob[{}] kappa:{}, acc:{}'.format(i, kappa, acc)
    print
    return (mf_tr / n_round, mf_te / n_round)


def make_mf_classification2(X ,y, clf, qid, X_test, n_round=3):
    print clf
    mf_tr = np.zeros((X.shape[0], 2))
    mf_te = np.zeros((X_test.shape[0], 2))
    for i in range(n_round):
        skf = StratifiedKFold(qid, n_folds=2, shuffle=True, random_state=42+i*1000)
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]

            y_tr = y[ind_tr]
            y_te = y[ind_te]

            clf.fit(X_tr, y_tr)
            try:
                mf_tr[ind_te] += clf.predict_proba(X_te)
                mf_te += clf.predict_proba(X_test)*0.5
            except:
                mf_tr[ind_te, 0] += clf.decision_function(X_te)
                mf_te[:, 0] += clf.decision_function(X_test)*0.5
                
            y_pred = np.round(clf.predict(X_te))
            kappa = pykappa.quadratic_weighted_kappa(y_te, y_pred)
            acc = np.mean(y_te == y_pred)
            print 'prob[{}] kappa:{}, acc:{}'.format(i, kappa, acc)
    print
    return (mf_tr / n_round, mf_te / n_round)


def learn_class_separators(clf, X_1, X_2, n_round):
    class_sep_4 = make_mf_classification2(X_1, (y_base<4), clf, q_base, X_2, n_round)
    class_sep_3 = make_mf_classification2(X_1, (y_base<3), clf, q_base, X_2, n_round)
    class_sep_2 = make_mf_classification2(X_1, (y_base<2), clf, q_base, X_2, n_round)
    class_sep_23 = make_mf_classification2(X_1, (y_base<4)*(y_base>1), clf, q_base, X_2, n_round)
    ret_tr = np.hstack((class_sep_4[0], class_sep_3[0], class_sep_2[0], class_sep_23[0]))
    ret_te = np.hstack((class_sep_4[1], class_sep_3[1], class_sep_2[1], class_sep_23[1]))
    return (ret_tr[:, 1::2], ret_te[:, 1::2])


logging.info("Assing names to featues")
X_base_tf = X_all_tr
X_hold_tf = X_all_te
X_base_add = X_additional_tr
X_hold_add = X_additional_te
X_base_w2v = X_w2v_tr
X_hold_w2v = X_w2v_te
X_base_counts = X_counts_tr
X_hold_counts = X_counts_te
X_base_counts2 = X_counts2_tr
X_hold_counts2 = X_counts2_te
q_base = qid_tr
q_hold = qid_te
y_base = y_all_tr

logging.info("Learn metafeatures")
# make features
rf_add_sep = learn_class_separators(
    RandomForestClassifier(n_estimators=500, n_jobs=-1,criterion='entropy', random_state=42),
    np.hstack((X_base_add, X_base_counts2)), 
    np.hstack((X_hold_add, X_hold_counts2)), 
    n_round=5)

mfs_rf_reg = make_mf_sliced_regression(
    train_df, 
    test_df,
    RandomForestRegressor(n_estimators=500, max_features=0.3, random_state=42), 
    n_round=3)

mfs_rf_clf = make_mf_sliced_regression(
    train_df, 
    test_df,
    RandomForestClassifier(n_estimators=500, max_features=0.3, random_state=42), 
    n_round=3)

mf_lsvc_clf = make_mf_classification4(
    X_base_tf, 
    y_base, 
    LinearSVC(), 
    q_base, 
    X_hold_tf, 
    n_round=10)

mf_lsvr_reg = make_mf_regression(
    X_base_tf, 
    y_base, 
    LinearSVR(), 
    q_base, 
    X_hold_tf, 
    n_round=10)


logging.info("Assemble 2nd level features")
X_train = np.hstack(
    (X_base_add, 
     mf_lsvr_reg[0][:, np.newaxis],
     mf_lsvc_clf[0],
     rf_add_sep[0],
     mfs_rf_clf[0][:, np.newaxis],
     mfs_rf_reg[0][:, np.newaxis],
     X_base_counts, 
     X_base_counts2))

X_test = np.hstack(
    (X_hold_add, 
     mf_lsvr_reg[1][:, np.newaxis],
     mf_lsvc_clf[1],
     rf_add_sep[1],
     mfs_rf_clf[1][:, np.newaxis],
     mfs_rf_reg[1][:, np.newaxis],
     X_hold_counts, 
     X_hold_counts2))

logging.info("Fit 2nd level model")
rfR = RandomForestRegressor(n_estimators=25000, n_jobs=-1, min_samples_split=3, random_state=42)
rfR.fit(X_train, y_base)
y_pred_rfR = rfR.predict(X_test)

logging.info("Dumping prediction")
np.savetxt(cfg.path_processed + 'mikhail_model2.txt', y_pred_rfR)

logging.info("Done!")