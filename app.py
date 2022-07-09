import pickle
from ast import literal_eval

import matplotlib
import pandas as pd
from flask import Flask, request, render_template

from utils.preprocess_web import WebPreprocessingClient
from utils.visualization import DataVisualization

app = Flask(__name__, template_folder="template")
matplotlib.use('Agg')

with open('model/model_2_1.pkl', 'rb') as f:
    model = pickle.load(f)

preproc = WebPreprocessingClient()
dataset = pd.read_csv("data/data-case-2.csv", converters={'tweet_proc': literal_eval})
visualization = DataVisualization(dataset)

piechart_url = visualization.generate_piechart()
wordcloud_url = visualization.generate_wordcloud(2)
wordcloud_neg_url = visualization.generate_wordcloud(0)
wordcloud_pos_url = visualization.generate_wordcloud(1)
neg_barplot_url = visualization.generate_barplot(0, model.freqs)
pos_barplot_url = visualization.generate_barplot(1, model.freqs)
classification_report_url = visualization.generate_classification_report(model.nbc_classification_report(0))
confusion_matrix_url = visualization.generate_confusion_matrix(model.nbc_confusion_matrix())


@app.route('/')
def root_page():
    return render_template("index.html",
                           piechart_url=piechart_url,
                           wordcloud_url=wordcloud_url,
                           wordcloud_neg_url=wordcloud_neg_url,
                           wordcloud_pos_url=wordcloud_pos_url,
                           neg_barplot_url=neg_barplot_url,
                           pos_barplot_url=pos_barplot_url,
                           classification_report_url=classification_report_url,
                           confusion_matrix_url=confusion_matrix_url,
                           predict_calc_prior=model.log_prior)


@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    tweet_proc = preproc.preprocess(tweet)
    predict_score = model.predict_naive_bayes(tweet_proc)
    words_likelihood = model.likelihoods_words(tweet_proc)

    predict_polarity = "Positive" if predict_score >= 0 else "Negative"
    return render_template("index.html",
                           piechart_url=piechart_url,
                           wordcloud_url=wordcloud_url,
                           wordcloud_neg_url=wordcloud_neg_url,
                           wordcloud_pos_url=wordcloud_pos_url,
                           neg_barplot_url=neg_barplot_url,
                           pos_barplot_url=pos_barplot_url,
                           classification_report_url=classification_report_url,
                           confusion_matrix_url=confusion_matrix_url,
                           predict_calc_prior=model.log_prior,
                           predict_tweet=tweet,
                           preproc_tweet=tweet_proc,
                           predict_score=predict_score,
                           words_likelihood=words_likelihood,
                           predict_polarity=predict_polarity)


if __name__ == "__main__":
    app.run(debug=True)
