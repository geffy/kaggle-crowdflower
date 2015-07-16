Kaggle ['Search Results Relevance'](https://www.kaggle.com/c/crowdflower-search-relevance)  2nd place solution
=======
### Mikhail Trofimov, Stanislav Semenov, Dmitry Altukhov

Gets score 0.71881 on private leaderboard

How to reproduce submission
=======
Don't forget to check paths in `./cfg.py`!
By default, you should place raw data in `./raw/`, create `./processed/` for temporal files and `./submit/` for submission file.

After this, run
```
python preprocessing_mikhail.py
python preprocessing_dmitry.py
python preprocessing_stanislav.py
python fit_models_dmitry.py
python fit_model1_mikhail.py
python fit_model2_mikhail.py
python blend.py
```

Dependencies
=======
* python 2.7.6
* numpy 1.10
* pandas 0.16.0
* scikit-learn 0.16.1
* scipy 0.15.1
* nltk 3.0.3
* BeautifulSoup 4.3.2
* tsne 0.1.1 (https://github.com/danielfrg/tsne)
* gensim 0.11.1-1
* backports.lzma 0.0.3

Hardware
=======
This code was developed on a machine with 32 cores and 60 gb ram (amazon cc2.8xlarge instance), however it is possible to build a solution even with the average laptop.
