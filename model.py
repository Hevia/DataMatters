from gensim.models import Word2Vec
from datasets import load_dataset

#load dataset
dataset = load_dataset("nedjmaou/MLMA_hate_speech")

# put data in list format
tweets = dataset['train']['tweet']

#train and save model
model = Word2Vec(tweets)
model.save("W2V_hate_speech.model")

#get most similar words? not sure of the usage for this
top10_most_similar = model.similar_by_word()
top20_most_similar = model.similar_by_word(20)
top_20_most_similar_to_hello = model.similar_by_word('hello', 20)