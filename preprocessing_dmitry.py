# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import cfg
from utility import *

train = pd.read_csv(cfg.path_train).fillna("")
test  = pd.read_csv(cfg.path_test ).fillna("")

X1, weights, titles, queries = assemble_counts(train,m='train')
X1_test, titles_test, queries_test = assemble_counts(test,m='test')
np.savetxt(cfg.path_features + 'train_counts.txt',X1)
np.savetxt(cfg.path_features +'test_counts.txt',X1_test)
pd.DataFrame(weights,columns=['weights']).to_csv(cfg.path_features + 'weights.csv',index=False)
pd.DataFrame(titles,columns=['titles_clean']).to_csv(cfg.path_features + 'titles_clean.csv',index=False)
pd.DataFrame(queries,columns=['queries_clean']).to_csv(cfg.path_features + 'queries_clean.csv',index=False)
pd.DataFrame(titles_test,columns=['titles_test_clean']).to_csv(cfg.path_features + 'titles_test_clean.csv',index=False)
pd.DataFrame(queries_test,columns=['queries_test_clean']).to_csv(cfg.path_features + 'queries_test_clean.csv',index=False)

#Extended queries top 10 words
train_ext, test_ext = construct_extended_query(queries,queries_test,titles,titles_test,top_words=10)
X5, queries_ext = assemble_counts2(train_ext.fillna(""))
X5_test, queries_ext_test = assemble_counts2(test_ext.fillna(""))
np.savetxt(cfg.path_features + 'train_ext_counts_top10.txt',X5)
np.savetxt(cfg.path_features + 'test_ext_counts_top10.txt',X5_test)
tmp = pd.DataFrame(train_ext,columns=['id','query','product_title','product_description','median_relevance','relevance_variance'])
tmp.to_csv(cfg.path_features + 'train_ext_top10.csv',index=False)
tmp = pd.DataFrame(test_ext,columns=['id','query','product_title','product_description'])
tmp.to_csv(cfg.path_features +  'test_ext_top10.csv',index=False)

#Extended queries top 15 words
train_ext, test_ext = construct_extended_query(queries,queries_test,titles,titles_test,top_words=15)
X5, queries_ext = assemble_counts2(train_ext.fillna(""))
X5_test, queries_ext_test = assemble_counts2(test_ext.fillna(""))
np.savetxt(cfg.path_features + 'train_ext_counts_top15.txt',X5)
np.savetxt(cfg.path_features + 'test_ext_counts_top15.txt',X5_test)
tmp = pd.DataFrame(train_ext,columns=['id','query','product_title','product_description','median_relevance','relevance_variance'])
tmp.to_csv(cfg.path_features + 'train_ext_top15.csv',index=False)
tmp = pd.DataFrame(test_ext,columns=['id','query','product_title','product_description'])
tmp.to_csv(cfg.path_features + 'test_ext_top15.csv',index=False)