import re

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk import word_tokenize
from nltk.corpus import stopwords

from utils.__const import custom_stopwords

cachedStopWords = stopwords.words("indonesian")
cachedStopWords.extend(custom_stopwords)


class WebPreprocessingClient:

    def __init__(self):
        self.colloquial = pd.read_csv("data/oth/colloquial-indonesian-lexicon-custom.csv")

    def __clean_text(self, tweet):
        tweet = tweet.lower()
        tweet = re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' ', tweet)
        tweet = re.sub(r'(#[a-z0-9]+)\w+', ' ', tweet)
        tweet = re.sub(r'(http\S+)', ' ', tweet)
        tweet = re.sub(r'\b(amp)\W', '', tweet)
        tweet = re.sub(r'[^a-zA-Z ]+?', ' ', tweet)
        tweet = re.sub(r'\b(?:a*(?:ha)+h?)\b', ' ', tweet)
        tweet = re.sub(r'\b(?=\w*(wk))\w+\b', ' ', tweet)
        tweet = re.sub(r'\b((moto)\s(gp))\b', 'motogp', tweet)
        tweet = re.sub('\W[a-z]\s', ' ', tweet)
        tweet = re.sub('\d+', '', tweet)
        tweet = re.sub('\s+', ' ', tweet)
        return tweet

    def __normalize_tweets(self, tweet):
        s = self.colloquial.set_index('slang')['formal'].to_dict()
        return ' '.join([s.get(i, i) for i in tweet.split()])

    def __remove_stopwords(self, tweet):
        return ' '.join([word for word in tweet.split() if word not in cachedStopWords])

    def __stem_tweets(self, tweet):
        return StemmerFactory().create_stemmer().stem(tweet)

    def __tokenize_tweets(self, tweet):
        return word_tokenize(tweet)

    def preprocess(self, tweet):
        tweet = self.__clean_text(tweet)
        tweet = self.__normalize_tweets(tweet)
        tweet = self.__remove_stopwords(tweet)
        tweet = self.__stem_tweets(tweet)
        tweet = self.__tokenize_tweets(tweet)
        return tweet
