from gensim.models import word2vec
import pandas as pd

#reading data
Dataset =pd.read_csv('ic.csv',engine='python',na_values=['NA','?'])
