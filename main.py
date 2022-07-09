import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
from ast import literal_eval
from builtins import set
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import seaborn as sns
import nbc
import utils.__const
from nbc import cNaiveBayesClassifier
from utils.label import LabellingClient
from utils.preprocess import PreprocessingClient
from utils.preprocess_web import WebPreprocessingClient
from utils.scrap import ScrapperClient
from utils.__const import custom_stopwords

print("======================= SCRAPPING =======================")
# scrapperClient = ScrapperClient()
# scrapperClient.get_tweets(query="mandalika", count=3000)
# scrapperClient.save_csv("../data/data-eval.csv")

print("===================== PREPROCESSING =====================")
# preprocessingClient = PreprocessingClient("data/eval/data-eval.csv")
# preprocessingClient.clean_text()
# preprocessingClient.remove_duplicates()
# preprocessingClient.normalize_tweets()
# preprocessingClient.remove_stopwords()
# preprocessingClient.stem_tweets()
# preprocessingClient.tokenize_tweets()
# preprocessingClient.remove_blank_tweets()
# preprocessingClient.save_csv("data/eval/data-eval-processed.csv")

print("======================= LABELLING =======================")
# labellingClient = LabellingClient("data/eval/data-eval-processed.csv")
# labellingClient.label_dataset()
# labellingClient.save_csv("data/eval/data-eval-labelled.csv")

print("========================= NBC ===========================")
# dataset1 = pd.read_csv("data/data-case-1.csv", converters={'tweet_proc': literal_eval})
# dataset2 = pd.read_csv("data/data-case-2.csv", converters={'tweet_proc': literal_eval})
# dataset3 = pd.read_csv("data/data-case-3.csv", converters={'tweet_proc': literal_eval})
# dataset4 = pd.read_csv("data/data-case-4.csv", converters={'tweet_proc': literal_eval})

# print("================ Model Training ====================")
# nbc = cNaiveBayesClassifier(dataset4, 3)
# print(nbc.nbc_classification_report())
# print(nbc.nbc_confusion_matrix())
# pickle.dump(nbc, open('model/model_4_3.pkl', 'wb'))

print("===================== EVALUASI ==========================")
# dataset = pd.read_csv("data/eval/data-eval-processed.csv")
# print(dataset[dataset['tweet_proc'] == "[]"])
#
# dataset = pd.read_csv("data/eval/data-eval-labelled.csv", converters={'tweet_proc': literal_eval})
# with open('model/model_2_1.pkl', 'rb') as f:
#     model = pickle.load(f)
#
# dataset['nbc'] = dataset['tweet_proc'].apply(lambda row: 1 if model.predict_naive_bayes(row) >= 0 else 0)
# neg_pred = 0
# pos_pred = 0
# for i in dataset['nbc']:
#     if i > 0:
#         pos_pred += 1
#     else:
#         neg_pred += 1
#
# print(pos_pred / (pos_pred + neg_pred) * 100)
# print(neg_pred / (pos_pred + neg_pred) * 100)
# dataset.to_csv("data/eval/data-eval-nbc.csv", sep=',', index=False)

print("================== VISUALISASI EVAL =====================")
dataset = pd.read_csv("data/eval/data-eval-nbc.csv")
polarity_count = [len(dataset[dataset['nbc'] == 0]),
                  len(dataset[dataset['nbc'] == 1])]

labels = ['Negative',
          'Positive']

centre_circle = plt.Circle((0, 0), 0.70, fc='white')

fig = plt.gcf()
fig.gca().add_artist(centre_circle)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}%\n({v:d})'.format(p=pct, v=val)

    return my_autopct

plt.pie(polarity_count,
        labels=labels,
        colors=sns.color_palette('pastel'),
        autopct=make_autopct(polarity_count))

plt.show()