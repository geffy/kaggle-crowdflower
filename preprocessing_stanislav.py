import cfg
import numpy as np
import pandas as pd

import random
import time
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

newdtrain = pd.read_pickle(cfg.path_processed + 'train_df')
newdtest = pd.read_pickle( cfg.path_processed + 'test_df')


def stasnormal(curstr):
    curstrstas = [' ']
    
    for j in range(len(curstr)):
        if curstr[j] in " abcdefghijklmnopqrstuvwxyz":
            curstrstas.append(curstr[j])
        
        if j == len(curstr) - 1:
            curstrstas.append(' ')
        
        if len(curstrstas) > 1 and curstrstas[-1] == ' ' and curstrstas[-2] == ' ':
            curstrstas = curstrstas[:-1]
    
    return ''.join(curstrstas)

querystemstas = []
titlestemstas = []
descstemstas = []

for i in range(len(newdtrain)):
    querystemstas.append(stasnormal(newdtrain['query_stem'][i]))
    titlestemstas.append(stasnormal(newdtrain['title_stem'][i]))
    descstemstas.append(stasnormal(newdtrain['desc_stem'][i]))
    
    if i % 1000 == 0:
        print i
    
newdtrain['query_stemstas'] = querystemstas
newdtrain['title_stemstas'] = titlestemstas
newdtrain['desc_stemstas'] = descstemstas

querystemstas = []
titlestemstas = []
descstemstas = []

for i in range(len(newdtest)):
    querystemstas.append(stasnormal(newdtest['query_stem'][i]))
    titlestemstas.append(stasnormal(newdtest['title_stem'][i]))
    descstemstas.append(stasnormal(newdtest['desc_stem'][i]))
    
    if i % 1000 == 0:
        print i
    
newdtest['query_stemstas'] = querystemstas
newdtest['title_stemstas'] = titlestemstas
newdtest['desc_stemstas'] = descstemstas

vect = TfidfVectorizer(
        strip_accents='unicode', analyzer='char',
        ngram_range=(1, 1), use_idf = 0).fit(newdtrain['title_stemstas'])

Xstats = np.zeros((len(newdtrain), 1))

for i in range(len(newdtrain)):
    query = newdtrain['query_stemstas'][i]
    
    Xstats[i] = len(query)

Xquery = vect.transform(newdtrain['query_stemstas']).todense()
Xtitle = vect.transform(newdtrain['title_stemstas']).todense()
Xdesc = vect.transform(newdtrain['desc_stemstas']).todense()

Xtrain = np.array((Xtitle + 100.) / (Xquery + 0.1) / Xstats)

Xstats = np.zeros((len(newdtest), 1))

for i in range(len(newdtest)):
    query = newdtest['query_stemstas'][i]
    
    Xstats[i] = len(query)

Xquery = vect.transform(newdtest['query_stemstas']).todense()
Xtitle = vect.transform(newdtest['title_stemstas']).todense()
Xdesc = vect.transform(newdtest['desc_stemstas']).todense()

Xtest = np.array((Xtitle + 100.) / (Xquery + 0.1) / Xstats)

np.savetxt(cfg.path_features + 'ssfeas4train.txt', Xtrain)
np.savetxt(cfg.path_features + 'ssfeas4test.txt', Xtest)