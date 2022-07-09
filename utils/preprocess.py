import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from utils.__const import custom_stopwords

cachedStopWords = stopwords.words("indonesian")
cachedStopWords.extend(custom_stopwords)


class PreprocessingClient:
    def __init__(self, filename):
        self.dataset = pd.read_csv(filename)
        self.colloquial = pd.read_csv("data/oth/colloquial-indonesian-lexicon-custom.csv")

    def clean_text(self):
        print("Cleaning text...")
        self.dataset['tweet_proc'] = self.dataset['tweet'] \
            .str.lower() \
            .str.replace(r'(^|[^@\w])@(\w{1,15})\b', ' ', regex=True) \
            .str.replace(r'(#[a-z0-9]+)\w+', ' ', regex=True) \
            .str.replace(r'(http\S+)', ' ', regex=True) \
            .str.replace(r'\b(amp)\W', '', regex=True) \
            .str.replace(r'[^a-zA-Z ]+?', ' ', regex=True) \
            .str.replace(r'\b(?:a*(?:ha)+h?)\b', ' ', regex=True) \
            .str.replace(r'\b(?=\w*(wk))\w+\b', ' ', regex=True) \
            .str.replace(r'\b((moto)\s(gp))\b', 'motogp', regex=True) \
            .str.replace(r'\W[a-z]\s', ' ', regex=True) \
            .str.replace(r'\d+', '', regex=True) \
            .str.replace(r'\s+', ' ', regex=True) \
            .str.strip()

    def remove_duplicates(self):
        print("Removing duplicates...")
        self.dataset = self.dataset.drop_duplicates(subset='tweet_proc', inplace=False, keep="first")

    def remove_blank_tweets(self):
        print("Removing blank tweets...")
        self.dataset = self.dataset[self.dataset['tweet_proc'].map(lambda d: len(d)) > 0]

    def normalize_tweets(self):
        print("Normalizing words...")
        s = self.colloquial.set_index('slang')['formal'].to_dict()
        s = {r"\b{}\b".format(k): v for k, v in s.items()}
        self.dataset['tweet_proc'] = self.dataset['tweet_proc'].replace(s, regex=True)

    def remove_stopwords(self):
        print("Removing stopwords...")
        self.dataset['tweet_proc'] = self.dataset['tweet_proc'].apply(
            lambda row: ' '.join([word for word in row.split() if word not in cachedStopWords]))

    def stem_tweets(self):
        print("Stemming...")
        stemmer = StemmerFactory().create_stemmer()
        self.dataset['tweet_proc'] = self.dataset['tweet_proc'].apply(lambda row: stemmer.stem(row))

    def tokenize_tweets(self):
        print("Tokenization....")
        self.dataset['tweet_proc'] = self.dataset.apply(lambda row: word_tokenize(row['tweet_proc']), axis=1)

    def save_csv(self, filename):
        print("Saving CSV...")
        self.dataset.to_csv(filename, sep=',', index=False)
