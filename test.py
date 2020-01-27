from gensim.models import Word2Vec


model = Word2Vec.load("word2vec.model")
Molecule2vec= model[model.wv.vocab]
total_exemples=model.corpus_count


print(model)
print(total_exemples)
print(Molecule2vec.shape)
print(Molecule2vec)


print('vector of c ',model.wv['c'])
print('vector of C ',model.wv['C'])
print('vector of O ',model.wv['O'])
print('vector of N ',model.wv['N'])




