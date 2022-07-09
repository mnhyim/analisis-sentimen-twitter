import base64
import io
from ast import literal_eval

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import STOPWORDS, WordCloud

from utils.__const import custom_stopwords

sns.set_style('darkgrid')
sns.color_palette('pastel')


class DataVisualization:
    def __init__(self, data):
        self.img = io.BytesIO()
        self.dataset = data

    def generate_piechart(self):
        polarity_count = [len(self.dataset[self.dataset['polarity'] == 0]),
                          len(self.dataset[self.dataset['polarity'] == 1])]
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

        plt.savefig(self.img, format='png')
        plt.close()
        self.img.seek(0)
        return base64.b64encode(self.img.getvalue()).decode('utf8')

    def generate_wordcloud(self, polarity):
        wc_stopwords = set(STOPWORDS)
        wc_stopwords.update(custom_stopwords)

        if polarity == 0:
            words = self.dataset[self.dataset['polarity'] == 0]['tweet_proc'].apply(lambda row: " ".join(row))
            words = list(words)
            words = pd.Series(words).str.cat(sep=' ')
        elif polarity == 1:
            words = self.dataset[self.dataset['polarity'] == 1]['tweet_proc'].apply(lambda row: " ".join(row))
            words = list(words)
            words = pd.Series(words).str.cat(sep=' ')
        else:
            words = self.dataset['tweet_proc'].apply(lambda row: " ".join(row))
            words = list(words)
            words = pd.Series(words).str.cat(sep=' ')

        wordcloud = WordCloud(stopwords=wc_stopwords,
                              height=400,
                              width=1000,
                              max_font_size=200,
                              max_words=50,
                              collocations=False,
                              background_color='black').generate(words)

        wordcloud.to_image().save(self.img, 'png')
        self.img.seek(0)

        return base64.b64encode(self.img.getvalue()).decode('utf8')

    def generate_barplot(self, pol, freqs):
        plt.figure(figsize=(8, 4), tight_layout=True)

        if pol == 0:
            neg_freq_list = sorted([[i[0][0], i[1]] for i in freqs.items() if i[0][1] == 0],
                                   key=lambda x: x[1],
                                   reverse=True)[:10]
            ax = sns.barplot(x=[i[0] for i in neg_freq_list], y=[i[1] for i in neg_freq_list])
            ax.set(xlabel="Kata", ylabel="Frekuensi")
            ax.bar_label(ax.containers[0])
            plt.savefig(self.img, format='png')
            plt.close()
            self.img.seek(0)
        else:
            pos_freq_list = sorted([[i[0][0], i[1]] for i in freqs.items() if i[0][1] == 1],
                                   key=lambda x: x[1],
                                   reverse=True)[:10]
            ax = sns.barplot(x=[i[0] for i in pos_freq_list], y=[i[1] for i in pos_freq_list])
            ax.set(xlabel="Kata", ylabel="Frekuensi")
            ax.bar_label(ax.containers[0])
            plt.savefig(self.img, format='png')
            plt.close()
            self.img.seek(0)

        return base64.b64encode(self.img.getvalue()).decode('utf8')

    def generate_classification_report(self, creport):
        sns.heatmap(pd.DataFrame(creport).iloc[:-1, :].T,
                    annot=True,
                    cmap="YlGnBu")
        plt.savefig(self.img, format='png')
        plt.close()
        self.img.seek(0)

        return base64.b64encode(self.img.getvalue()).decode('utf8')

    def generate_confusion_matrix(self, cmatrix):
        group_names = ['True-Neg', 'False-Pos', 'False-Neg', 'True-Pos']
        group_counts = ['{0: 0.0f}'.format(value) for value in cmatrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cmatrix.flatten() / np.sum(cmatrix)]

        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        sns.heatmap(cmatrix / np.sum(cmatrix),
                    annot=labels,
                    fmt='',
                    cmap='Blues')

        plt.savefig(self.img, format='png')
        plt.close()
        self.img.seek(0)

        return base64.b64encode(self.img.getvalue()).decode('utf8')
