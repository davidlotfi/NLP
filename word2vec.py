from gensim.models import Word2Vec
import pandas as pd

#reading data
Dataset =pd.read_csv('ic.csv',sep=';',engine='python',na_values=['NA','?'])

# Molecules
#print(Dataset.iloc[:,7])

Molecule= Dataset.iloc[:,7]
#print(Molecule[7])

#Build model vocabulary for molecules ( model of transormation word >>> vecteur )

model = Word2Vec(size=10,min_count=1)
model.build_vocab(Molecule)

total_exemples=model.corpus_count
print(total_exemples)

SmilesCanonical2 = model[model.wv.vocab]
print(SmilesCanonical2)
