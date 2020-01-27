from gensim.models import Word2Vec
import pandas as pd

#reading data
Dataset = pd.read_csv('ic.csv',sep=';',engine='python',na_values=['NA','?'])

# Molecules colonne
Molecule= Dataset.iloc[:,7].values

#print(Molecule)
#print(Molecule.shape)

#Build model vocabulary for molecules ( model of transormation word >>> vecteur )

model = Word2Vec(size=1,min_count=1)
model.build_vocab(Molecule)
model.save("word2vec.model")

total_exemples=model.corpus_count
print(model)
print(total_exemples)

Molecule2vec= model[model.wv.vocab]
#print(Molecule2vec)
print(Molecule2vec.shape)
print(Molecule2vec)
#model.train(Molecule2vec,total_examples=10, epochs=10)

# en va imprimer la vecteur de chaque atom exp( N = ?)

#c = model.wv['c']
#print(c)

