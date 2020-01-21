from gensim.models import word2vec
import pandas as pd

#reading data
Dataset =pd.read_csv('ic.csv',sep=';',engine='python',na_values=['NA','?'])

# Molecules
#print(Dataset.iloc[:,7])

Molecule= Dataset.iloc[:,7]
print(Molecule[7])