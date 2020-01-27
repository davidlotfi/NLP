from gensim.models import Word2Vec


wordarray = ['a','b','d']

model = Word2Vec(size=1,min_count=1)
model.build_vocab(wordarray)
total_examples = model.corpus_count

print(model)
print(total_examples)
