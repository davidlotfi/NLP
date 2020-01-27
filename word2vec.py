from gensim.models import Word2Vec
import pandas as pd

#reading data

def dataManipulation():
     Dataset = pd.read_csv('ic.csv',sep=';',engine='python',na_values=['NA','?'])
     # Molecules colonne
     Molecule= Dataset.iloc[:,7].values
     return Molecule


#print(Molecule.shape)

#Build model vocabulary for molecules ( model of transormation word >>> vecteur )
def modele():
    model = Word2Vec(size=1,min_count=1)
    model.build_vocab(dataManipulation())
    model.save("word2vec.model")
    return model




print(dataManipulation())
print(modele())




