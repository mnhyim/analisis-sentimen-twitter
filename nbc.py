from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


class cNaiveBayesClassifier:
    def __init__(self, dataset, type):
        self.dataset = dataset

        self.training_data, self.testing_data = self.split_dataset(type)
        self.freqs = self.create_freq_dict(self.training_data["tweet_proc"],
                                           self.training_data["polarity"])
        self.log_prior, self.log_likelihood = self.train_naive_bayes(self.freqs,
                                                                     self.training_data["tweet_proc"],
                                                                     self.training_data["polarity"])

    def split_dataset(self, type):
        if type == 1:
            training_data = self.dataset.sample(frac=0.90, random_state=24)
        elif type == 2:
            training_data = self.dataset.sample(frac=0.80, random_state=24)
        else:
            training_data = self.dataset.sample(frac=0.70, random_state=24)
        testing_data = self.dataset.drop(training_data.index)

        return training_data, testing_data

    def create_freq_dict(self, x_train, y_train):
        ys_list = np.squeeze(y_train).tolist()
        freqs = {}

        for y, tweet in zip(ys_list, x_train):
            for word in tweet:
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1

        return freqs

    def lookup_word(self, freqs, word, label):
        n = 0
        pair = (word, label)
        if pair in freqs:
            n = freqs[pair]

        return n

    def train_naive_bayes(self, freqs, train_x, train_y):
        log_likelihood = {}
        log_prior = 0

        vocab = set([pair[0] for pair in freqs.keys()])
        V = len(vocab)

        N_pos = N_neg = V_pos = V_neg = 0
        for pair in freqs.keys():
            if pair[1] > 0:
                V_pos += 1
                N_pos += freqs[pair]
            else:
                V_neg += 1
                N_neg += freqs[pair]

        D = len(train_y)
        D_neg = len(list(filter(lambda x: x == 0, train_y)))
        D_pos = len(list(filter(lambda x: x == 1, train_y)))
        # print(D)
        # print(D_neg)
        # print(D_pos)
        # print(np.log(D_neg))
        # print(np.log(D_pos))
        # print(np.log(D_pos) - np.log(D_neg))
        log_prior = np.log(D_pos) - np.log(D_neg)

        for word in vocab:
            freqs_pos = self.lookup_word(freqs, word, 1)
            freqs_neg = self.lookup_word(freqs, word, 0)

            p_w_pos = (freqs_pos + 1) / (N_pos + V)
            p_w_neg = (freqs_neg + 1) / (N_neg + V)

            log_likelihood[word] = np.log(p_w_pos / p_w_neg)

        return log_prior, log_likelihood

    def predict_naive_bayes(self, tweet):
        p = 0
        p += self.log_prior
        for word in tweet:
            if word in self.log_likelihood:
                p += self.log_likelihood[word]

        return p

    def test_naive_bayes(self):
        res = []
        data = self.testing_data
        for i in data["tweet_proc"]:
            p = self.predict_naive_bayes(i)
            if p >= 0:
                res.append(1)
            else:
                res.append(0)

        data["nbc_pred"] = res
        return data

    def nbc_classification_report(self, type=1):
        test_res = self.test_naive_bayes()
        if type == 0:
            return classification_report(test_res["polarity"],
                                         test_res["nbc_pred"],
                                         target_names=["Negative", "Positive"],
                                         output_dict=True)
        else:
            return classification_report(test_res["polarity"],
                                         test_res["nbc_pred"],
                                         target_names=["Negative", "Positive"])

    def nbc_confusion_matrix(self):
        test_res = self.test_naive_bayes()
        return confusion_matrix(test_res["polarity"], test_res["nbc_pred"])

    def likelihoods_words(self, tweet):
        res = []
        for word in tweet:
            if word in self.log_likelihood:
                res.append(f"{word}, {self.log_likelihood[word]}")
        return res
