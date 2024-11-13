from gensim.models import Word2Vec
from datasets import load_dataset
import json

#load dataset
dataset = load_dataset("nedjmaou/MLMA_hate_speech")

with open('./data/hate_speech.json', 'r', encoding="utf-8") as json_file:
    hate_speech = json.load(json_file)

# put data in list format
tweets = dataset['train']['tweet']

#train and save model
model = Word2Vec(sentences=tweets, vector_size=100, window=5, min_count=1, workers=4)
model.save("W2V_hate_speech.model")
sims = model.wv.most_similar('computer', topn=10) 