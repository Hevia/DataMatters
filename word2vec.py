from gensim.models import Word2Vec
from datasets import load_dataset
import json
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd

#load dataset
dataset = load_dataset("nedjmaou/MLMA_hate_speech")

with open('./data/hate_speech.json', 'r', encoding="utf-8") as json_file:
    hate_speech = json.load(json_file)

# put data in list format
tweets = dataset['train']['tweet']
# Download stopwords
nltk.download('stopwords')

# Define stop words and punctuation
stop_words = set(stopwords.words('english')).union(set(stopwords.words('arabic'))).union(set(stopwords.words('french')))
punctuation = set(string.punctuation)
custom_stop_words = set(["user", "y'all", "ur", "url", "ok", "oh", "lol", "lmao", "....."])
remove_words = stop_words.union(punctuation).union(custom_stop_words)

# Tokenize tweets and remove stop words and punctuation
tokenized_tweets = [
    [word for word in nltk.word_tokenize(tweet.lower()) if word not in remove_words and word not in punctuation]
    for tweet in tweets
]

#train and save model
model = Word2Vec(sentences=tokenized_tweets, vector_size=100, window=5, min_count=1, workers=6, max_vocab_size=30000)
model.save("./output/W2V_hate_speech.model")

def grab_most_similar(model, word, n=10):
    try:
        sims = model.wv.most_similar(word, topn=n)
        return [result[0] for result in sims]
    except KeyError:
        return []

results_dict = {}

for lang in hate_speech:
    for word in hate_speech[lang]:
        results_dict[word] = grab_most_similar(model, word)

# Save the dict to csv
# Convert the results_dict to a DataFrame
df = pd.DataFrame(list(results_dict.items()), columns=['word', 'most_similar_words'])

# Join the list of similar words into a single string
df['most_similar_words'] = df['most_similar_words'].apply(lambda x: ', '.join(x))

# Save the DataFrame to a CSV file
df.to_csv('./output/most_similar_words.csv', index=False, encoding='utf-8')