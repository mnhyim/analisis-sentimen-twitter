from ast import literal_eval

import pandas as pd


class LabellingClient:
    def __init__(self, filename):
        self.data = pd.read_csv(filename, converters={'tweet_proc': literal_eval})

        self.neg_lexicon = pd.read_table("data/oth/negative-lexicon.tsv")
        self.pos_lexicon = pd.read_table("data/oth/positive-lexicon.tsv")

        self.neg_lexicon_dict = dict(zip(self.neg_lexicon['word'], self.neg_lexicon['weight']))
        self.pos_lexicon_dict = dict(zip(self.pos_lexicon['word'], self.pos_lexicon['weight']))

    def determine_polarity(self, sentences):
        score = 0

        for word in sentences:
            for key, value in self.neg_lexicon_dict.items():
                if word == key:
                    score += int(value)

        for word in sentences:
            for key, value in self.pos_lexicon_dict.items():
                if word == key:
                    score += int(value)

        return score, 1 if score >= 0 else 0

    def label_dataset(self):
        print("Labelling...")

        res = list(zip(*self.data['tweet_proc'].apply(lambda row: self.determine_polarity(row))))
        self.data["score"] = res[0]
        self.data["polarity"] = res[1]

        print(self.data['polarity'].value_counts(normalize=True))

    def save_csv(self, filename):
        print("Saving CSV...")
        self.data.to_csv(filename, sep=',', index=False)
