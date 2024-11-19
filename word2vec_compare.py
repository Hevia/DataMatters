import gensim.downloader as api
import json
import pandas as pd

# get the hate speech words we're interested in
with open('./data/hate_speech.json', 'r', encoding="utf-8") as json_file:
    hate_speech = json.load(json_file)

# load 2 different pretrained word2vec models
# see https://github.com/piskvorky/gensim-data for details on the models
wv_glove = api.load('glove-twitter-200') 
wv_google = api.load('word2vec-google-news-300')

def grab_most_similar(model, word, n=10):
    try:
        sims = model.most_similar(word, topn=n)
        return [result[0] for result in sims]
    except KeyError:
        return []

results_dict = {}

for lang in hate_speech:
    for word in hate_speech[lang]:
        results_dict[word] = grab_most_similar(wv_glove, word) # replace the first param here with the model you're interested in comparing

# Save the dict to csv
# Convert the results_dict to a DataFrame
df = pd.DataFrame(list(results_dict.items()), columns=['word', 'most_similar_words'])

# Join the list of similar words into a single string
df['most_similar_words'] = df['most_similar_words'].apply(lambda x: ', '.join(x))

# Save the DataFrame to a CSV file
df.to_csv('./output/most_similar_words_glove_twitter_200.csv', index=False, encoding='utf-8') # don't forget to change the filename otherwise you'll overwrite your work