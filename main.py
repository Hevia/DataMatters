from lingua import Language, LanguageDetectorBuilder
from datasets import load_dataset
import nltk
import json
import pandas as pd

# Download punkt if we dont already have it for later tokenization
# nltk.download('punkt')
# nltk.download('punkt_tab') # this might be needed for arabic tokenization

# We use the lingua library to detect the language of the tweet
languages = [Language.ENGLISH, Language.FRENCH, Language.ARABIC]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

# The data is loaded using huggingface's datasets library
# which is a really popular choice in industry because of how it efficiently loads data on disk
# bunch of dumb tech junk that you don't need to worry about, but its good to say you used it!
dataset = load_dataset("nedjmaou/MLMA_hate_speech")

# put data in list format
tweets = dataset['train']['tweet']

#load the list of hate speech words into a nested dictionary
with open('./data/hate_speech.json', 'r', encoding="utf-8") as json_file:
    hate_speech = json.load(json_file)

#create dictionary with counts of tweets containing several categories of speech
hate_speech_counts = {}
total_tweets = 0

# do a basic keyword search for hate speech in the tweets
for tweet in tweets:
    # detect language and add to counts if not already present
    language = detector.detect_language_of(tweet).name.lower()
    if language not in hate_speech_counts:
        hate_speech_counts[language] = {
            'total': 0,
            'total_hate': 0,
            'target': {},
            'sentiment': {}
        }
    
    # normalize and tokenize tweets
    tokens = nltk.word_tokenize(tweet.lower())

    # Increment tweet count
    total_tweets += 1
    hate_speech_counts[language]['total'] += 1

    # Iterate through the tokens and check if they are in the hate speech dictionary
    for token in tokens:
        # Check each language's hate speech dictionary
        for lang in hate_speech:
            entry = hate_speech[lang].get(token)
            if entry:
                hate_speech_counts[language]['total_hate'] += 1
                
                # Dynamically add target category if not exists
                target = entry['target']
                if target not in hate_speech_counts[language]['target']:
                    hate_speech_counts[language]['target'][target] = 0
                hate_speech_counts[language]['target'][target] += 1
                
                # Dynamically add sentiment category if not exists
                sentiment = entry['sentiment']
                if sentiment not in hate_speech_counts[language]['sentiment']:
                    hate_speech_counts[language]['sentiment'][sentiment] = 0
                hate_speech_counts[language]['sentiment'][sentiment] += 1


total_hate_speech_all = sum(hate_speech_counts[lang]['total_hate'] for lang in hate_speech_counts)
tweets_no_hate_speech = total_tweets - total_hate_speech_all

# Convert nested dictionaries to DataFrames for each language
for language in hate_speech_counts:
    # Create DataFrame for target categories
    target_df = pd.DataFrame.from_dict(hate_speech_counts[language]['target'], orient='index', columns=['count'])
    target_df.index.name = 'Target'  # Add Target header
    target_df.to_csv(f'./data/targets/{language}_counts.csv')
    
    # Create DataFrame for sentiment categories
    sentiment_df = pd.DataFrame.from_dict(hate_speech_counts[language]['sentiment'], orient='index', columns=['count'])
    sentiment_df.index.name = 'Sentiment'  # Add Sentiment header
    sentiment_df.to_csv(f'./data/sentiments/{language}_counts.csv')