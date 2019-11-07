# --------------------------------------------------------- #
############### EXTRACTING BITCOIN TWEET DATA ###############
# --------------------------------------------------------- #
import pandas as pd
from nltk.tokenize import RegexpTokenizer
# --------------- IMPORT LABELED LABEL DATA -------------------- #
# https://github.com/mjain72/Sentiment-Analysis-using-Word2Vec-and-LSTM

# There is no large data collection available for annotated financial tweets

tweetsData = pd.read_csv('datasets/Sentiment Analysis Dataset.csv', skiprows=[8835, 535881])
#skiping these two rows as they have some bad data
print(tweetsData.head())
import twitter

#Dividing the dataset into features and lables
tweets = tweetsData['SentimentText']
labels = tweetsData['Sentiment']

#check the distribution of lebels

labels_count = labels.value_counts()
labels_count.plot(kind="bar")
print(labels.value_counts())

#Looks like the distribution is even

#Lower and split the dialog
#and use regular expression to keep only letters we will use nltk Regular expression package
tkr = RegexpTokenizer('[a-zA-Z@]+')

tweets_split = []

for i, line in enumerate(tweets):
    #print(line)
    tweet = str(line).lower().split()
    tweet = tkr.tokenize(str(tweet))
    tweets_split.append(tweet)

print(tweets_split[1])

# Datasets from #
# https://www.kaggle.com/alaix14/bitcoin-tweets-20160101-to-20190329 #
# user
# fullname
# tweet-id
# timestamp
# url
# likes
# replies
# retweets
# text
# html
df = pd.read_csv('datasets/tweets.csv')
print(df)

print(df[['timestamp']])
df.iloc[100000,8]


