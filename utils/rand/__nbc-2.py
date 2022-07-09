import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
import numpy as np

dataset = pd.read_csv("../../data/__backups/backups/data-mandalika-labelled-1.csv", converters={'tweet_proc': literal_eval})

# drop score == 0
# dataset = dataset[dataset['score] != 0]
# print(dataset)

dataset['polarity'] = LabelEncoder().fit_transform(dataset['polarity'])
dataset['tweet_proc'] = dataset['tweet_proc'].apply(lambda row: " ".join(row))

x = dataset['tweet_proc']
y = dataset['polarity']
x, x_test, y, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

# TF-IDF
vec = TfidfVectorizer(lowercase=False, min_df=2)
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

model = MultinomialNB()
model.fit(x, y)

print("acc:", model.score(x_test, y_test))

# EVAL
eval = [
    'sirkuit jelek kotor rawat',
    'indah mandalika',
]
print(model.predict(vec.transform(eval)))
