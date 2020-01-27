from gensim.models import Word2Vec


wordarray = ['a','b','d']

model = Word2Vec(size=1,min_count=1)
model.build_vocab(wordarray)
wordV = model[model.wv.vocab]

total_examples = model.corpus_count

print(model)
print(wordV)
print(total_examples)


print('vector of a ',model.wv['a'])
print('vector of b ',model.wv['b'])
print('vector of d ',model.wv['d'])