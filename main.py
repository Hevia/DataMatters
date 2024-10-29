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

# The data is now in a list format, and the code to iterate through the list is prepared for you
tweets = dataset['train']['tweet']

# We load the list of hate speech words into a nested dictionary


# Puts some variables here to record whatever you think is necessary
# TODO

# We're going to do a basic keyword search for hate speech in the tweets
# Iterate through the tweets
for tweet in tweets:
    # You should probably first convert the tweet to lowercase as a form of normalization
    # TODO

    # Use the nltk tokenizer to tokenize the tweet
    # Massive hint: Just use nltk.word_tokenize(<lowercase variable of tweet here>)
    # TODO

    # Detect the language and record that result for later tracking
    # TODO

    # Iterate through the tokens and check if they are in the hate speech dictionary
    # There are several categories to check through
    # If they are, record it, else move on but record any misses
    # TODO

    pass # Remove this line when you start coding