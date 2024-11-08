from lingua import Language, LanguageDetectorBuilder
from datasets import load_dataset
import nltk
import json

# Download punkt if we dont already have it for later tokenization
nltk.download('punkt')
nltk.download('punkt_tab') # this might be needed for arabic tokenization

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
with open('./data/hate_speech.json', 'r') as json_file:
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
        'target': {
            'gender': 0,
            'origin': 0,
            'disability': 0,
            'sexual_orientation': 0,
            'race': 0,
            'religion': 0,
            'other': 0,
        },
        'sentiment': {
            'negative': 0,
            'neutral': 0,
        }
    }
    # normalize and tokenize tweets
    tokens = nltk.word_tokenize(tweet.lower())

    # Iterate through the tokens and check if they are in the hate speech dictionary
    for token in tokens:
        total_tweets += 1
        hate_speech_counts[language]['total'] += 1
        entry = hate_speech.get(language).get(token)
        if entry:
            hate_speech_counts[language]['total_hate'] += 1
            hate_speech_counts[language]['target'][entry['target']] += 1
            hate_speech_counts[language]['sentiment'][entry['sentiment']] += 1


total_hate_speech_all = hate_speech_counts['french']['total_hate'] + hate_speech_counts['english']['total_hate'] + hate_speech_counts['arabic']['total_hate']
tweets_no_hate_speech = total_tweets - total_hate_speech_all

#demo for proof of concept, delete at will
print("Total tweets containing hate speech: ", total_hate_speech_all )
print("Total tweets without hate speech: ", tweets_no_hate_speech)
print("English tweet hate speech data: ", hate_speech_counts['english'])
print("English tweets containing anti-lgbt hate speech: ", hate_speech_counts['english']['target']['sexual_orientation'])