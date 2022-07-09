import pandas as pd
from ast import literal_eval

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)


class LabellingClient:
    def __init__(self, filename):
        self.df = pd.read_csv(filename, converters={'tweet_proc': literal_eval})
        self.neg = pd.read_table("data/dll/negative-lexicon.tsv")
        self.pos = pd.read_table("data/dll/positive-lexicon.tsv")

        self.neg_dict = dict(zip(self.neg['word'], self.neg['weight']))
        self.pos_dict = dict(zip(self.pos['word'], self.pos['weight']))

    def __determine_polarity(self, sentences):
        score = 0

        for word in sentences:
            for key, value in self.neg_dict.items():
                if word == key:
                    score += int(value)

        for word in sentences:
            for key, value in self.pos_dict.items():
                if word == key:
                    score += int(value)

        return score, "positive" if score >= 0 else "negative"

    def label_dataset(self):
        print("Labelling...")

        res = list(zip(*self.df['tweet_proc'].apply(lambda row: self.__determine_polarity(row))))
        self.df["score"] = res[0]
        self.df["polarity"] = res[1]

        print()
        print(self.df['polarity'].value_counts(normalize=True))
        print()

    def save_csv(self, filename):
        print("Saving CSV...")
        self.df.to_csv(filename, sep=',', index=False)

