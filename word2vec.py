from gensim.models import Word2Vec
import pandas as pd

#reading data
Dataset = pd.read_csv('ic.csv',sep=';',engine='python',na_values=['NA','?'])

# Molecules
#print(Dataset.iloc[:,7])

Molecule= Dataset.iloc[:,7].values
print(Molecule.shape)

#Build model vocabulary for molecules ( model of transormation word >>> vecteur )

model = Word2Vec(size=1,min_count=1)
model.build_vocab(Molecule)
model.save("word2vec.model")
total_exemples=model.corpus_count
print(total_exemples)

Molecule2vec= model[model.wv.vocab]
print(Molecule2vec)
print(Molecule2vec.shape)
