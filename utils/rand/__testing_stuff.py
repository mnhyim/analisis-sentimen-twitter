from ast import literal_eval

import pandas as pd

from nbc import cNaiveBayesClassifier

# df = pd.read_csv("../../data/__backups/backups/data-raw.csv")
df = pd.read_csv("../../data/eval/bu/data-eval-labelled.csv", converters={'tweet_proc': literal_eval})
pd.set_option('display.max_columns', None)
print(df)
#
# nbc = cNaiveBayesClassifier(df,1)
# print(len(nbc.dataset))
# print(len(nbc.training_data))
# print(len(nbc.testing_data))