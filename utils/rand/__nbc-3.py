import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval


class NaiveBayesClassifier:
    def __init__(self, file):
        self.dataset = pd.read_csv(file, converters={'tweet_proc': literal_eval})
        self.dataset['polarity'] = LabelEncoder().fit_transform(self.dataset['polarity'])

        self.dataset_pos, self.dataset_neg = [d for _, d in self.dataset.groupby(['polarity'])]

        self.training_data = self.dataset.sample(frac=0.8, random_state=50)
        self.testing_data = self.dataset.drop(self.training_data.index)

    def calculate_prior_probability(self):
        return None

    def calc_likelihoods(self):
        return None


nbc = NaiveBayesClassifier("../data/__backups/data-mandalika-labelled-3.csv")
