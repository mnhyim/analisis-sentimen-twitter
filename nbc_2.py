from ast import literal_eval

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv("data/data-case-4.csv", converters={'tweet_proc': literal_eval})
x = dataset['tweet_proc'].apply(lambda row: " ".join(row))
y = dataset['polarity']

vect = CountVectorizer(binary=True)
x = vect.fit_transform(x).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

nb = MultinomialNB()
nb.fit(x_train, y_train)
print(nb.score(x_test, y_test))