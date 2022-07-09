import tweepy
import csv
from utils.__const import *


class ScrapperClient:
    def __init__(self):
        auth = tweepy.OAuthHandler(apiKey, apiKeySecret)
        self.api = tweepy.API(auth)
        self.tweets = []

    def get_tweets(self, query, count):
        print("Scrapping tweet..")
        print("query={q}, max={c}".format(q=query, c=count))

        for item in tweepy.Cursor(self.api.search_tweets,
                                  q="{q} -filter:retweets".format(q=query),
                                  tweet_mode="extended",
                                  count=count,
                                  lang="id").items():
            if len(self.tweets) == count:
                break
            self.tweets.append([item.id,
                                item.created_at.strftime("%d-%b-%Y (%H:%M:%S)"),
                                item.full_text.replace("\n", " ")])

        print("scrapped {t} tweets...".format(t=len(self.tweets)))

    def save_csv(self, filename):
        with open("data/{f}".format(f=filename), "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "date", "tweet"])
            writer.writerows(self.tweets)
