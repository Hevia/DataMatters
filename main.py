from lingua import Language, LanguageDetectorBuilder
from datasets import load_dataset
import nltk

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

# We load the list of hate speech words into a nested dictionary
# Currently has some fake stuff for demo, replace with real speech when available.
# Current code assumes that each entry is of form
    # word : {
        #'target' : < >,
        #'sentiment' : < >,
#       }
#possible options for target {gender, origin, disability, sexual_orientation, religion, other}
#possible options for sentiment are {positive, neutral, negative}
fake_hate_speech = {
    'hey' : {
        'target' : 'sexual_orientation',
        'sentiment' : 'neutral'
    },
    'buzzfeed' : {
        'target' : 'religion',
        'sentiment' : 'negative'
    },
    'oui' : {
        'target' : 'origin',
        'sentiment' : 'positive'
    }
}

#create dictionary with counts of tweets containing several categories of speech
hate_speech_counts = {
    'total_hate_all': 0,
    Language.ENGLISH:{
        'total' : 0,
        'total_hate': 0,
        'target': {
            'gender': 0,
            'origin' : 0,
            'disability' : 0,
            'sexual_orientation' : 0,
            'religion' : 0,
            'other': 0,
        },
        'sentiment':{
            'positive' : 0,
            'negative' : 0,
            'neutral' : 0,
        }
    },
    Language.FRENCH:{
        'total' : 0,
        'total_hate': 0,
        'target': {
            'gender': 0,
            'origin' : 0,
            'disability' : 0,
            'sexual_orientation' : 0,
            'religion' : 0,
            'other': 0,
        },
        'sentiment':{
            'positive' : 0,
            'negative' : 0,
            'neutral' : 0,
        }
    },
    Language.ARABIC:{
        'total' : 0,
        'total_hate': 0,
        'target': {
            'gender': 0,
            'origin' : 0,
            'disability' : 0,
            'sexual_orientation' : 0,
            'religion' : 0,
            'other': 0,
        },
        'sentiment':{
            'positive' : 0,
            'negative' : 0,
            'neutral' : 0,
        }
    }
}
total_tweets = 0

# do a basic keyword search for hate speech in the tweets
for tweet in tweets:
    # detect language
    language = detector.detect_language_of(tweet)

    # normalize and tokenize tweets
    tokens = nltk.word_tokenize(tweet.lower())

    # Iterate through the tokens and check if they are in the hate speech dictionary
    for token in tokens:
        total_tweets += 1
        hate_speech_counts[language]['total'] += 1
        entry = fake_hate_speech.get(token)
        if entry:
            hate_speech_counts['total_hate_all'] += 1
            hate_speech_counts[language]['total_hate'] += 1
            hate_speech_counts[language]['target'][entry['target']] += 1
            hate_speech_counts[language]['sentiment'][entry['sentiment']] += 1

tweets_no_hate_speech = total_tweets - hate_speech_counts['total_hate_all']

#demo for proof of concept, delete at will
print("Total tweets containing hate speech: ", hate_speech_counts['total_hate_all'])
print("Total tweets without hate speech: ", tweets_no_hate_speech)
print("French tweets: ", hate_speech_counts[Language.FRENCH]['total'])
print("French tweets containing hate speech: ", hate_speech_counts[Language.FRENCH]['total_hate'])
print("English tweets containing anti-lgbt hate speech: ", hate_speech_counts[Language.ENGLISH]['target']['sexual_orientation'])
