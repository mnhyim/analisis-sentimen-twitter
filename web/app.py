import base64
import io
from ast import literal_eval

import flask
import matplotlib
import pandas as pd
from flask import Flask
from flask import jsonify, request, render_template
from matplotlib import pyplot as plt
import seaborn as sns
from werkzeug.utils import redirect
from wordcloud import STOPWORDS, WordCloud
from nbc import cNaiveBayesClassifier

from utils.preprocessing import PreprocessingClient

matplotlib.use('Agg')
app = Flask(__name__, template_folder=".")

dataset = pd.read_csv("../data/__backups/backups/data-mandalika-labelled-1.csv", converters={'tweet_proc': literal_eval})
data_neg = dataset[dataset['polarity'] == 'negative']
data_pos = dataset[dataset['polarity'] == 'positive']
all_word = dataset['tweet_proc'].apply(lambda row: " ".join(row))

preprocessor = PreprocessingClient()


nbc = cNaiveBayesClassifier(dataset)
training_data, testing_data = nbc.split_dataset()
freqs = nbc.create_freq_dict(training_data["tweet_proc"], training_data["polarity"])
log_prior, log_likelihood = nbc.train_naive_bayes(freqs, training_data["tweet_proc"], training_data["polarity"])
img = io.BytesIO()


def generate_pie():
    colors = sns.color_palette('pastel')
    polarity_count = [len(data_neg), len(data_pos)]
    labels = ['Negative', 'Positive']

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.pie(polarity_count, labels=labels, colors=colors, autopct='%.0f%%')

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')


def generate_wordcloud(pol):
    stopwords = set(STOPWORDS)
    # stopwords.update(["mandalika"])
    if pol == 0:
        # WordCloud data negative
        all_word = data_neg['tweet_proc'].apply(lambda row: " ".join(row))
        all_word = list(all_word)
        all_word = pd.Series(all_word).str.cat(sep=' ')

    elif pol == 1:
        # WordCloud data positive
        all_word = data_pos['tweet_proc'].apply(lambda row: " ".join(row))
        all_word = list(all_word)
        all_word = pd.Series(all_word).str.cat(sep=' ')

    else:
        # WordCloud all data
        all_word = dataset['tweet_proc'].apply(lambda row: " ".join(row))
        all_word = list(all_word)
        all_word = pd.Series(all_word).str.cat(sep=' ')

    wordcloud = WordCloud(stopwords=stopwords, max_font_size=200, max_words=100,
                          collocations=False, background_color='black').generate(all_word)
    wordcloud.to_image().save(img, 'png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')


@app.route('/')
def root():
    return render_template('index.html', pie_url=generate_pie(), neg_cloud_url=generate_wordcloud(0),
                           pos_cloud_url=generate_wordcloud(1), cloud_url=generate_wordcloud(2))


@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    predict_res = nbc.naive_bayes_predict(tweet, log_prior, log_likelihood)
    return render_template('index.html', pie_url=generate_pie(), neg_cloud_url=generate_wordcloud(0),
                           pos_cloud_url=generate_wordcloud(1), cloud_url=generate_wordcloud(2), inputted_tweet=tweet,
                           predict_res=predict_res, polarity="Positive" if predict_res >= 0  else "Negative")


if __name__ == "__main__":
    app.run(debug=True)
