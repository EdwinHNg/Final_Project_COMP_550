# -----------------------------------
import gensim.models.keyedvectors as word2vec #need to use due to depreceated model
from nltk.tokenize import RegexpTokenizer

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve,  roc_auc_score, classification_report


# --------------------------------------------------------- #
############### EXTRACTING BITCOIN TWEET DATA ###############
# --------------------------------------------------------- #
# https://www.kaggle.com/alaix14/bitcoin-tweets-20160101-to-20190329

# ------------------------------------------------------------------- #
# --------------- IMPORT TWEETER LABEL DATA ------------------------- #
# https://github.com/mjain72/Sentiment-Analysis-using-Word2Vec-and-LSTM
# ------------------------------------------------------------------- #

# There is no large data collection available for annotated financial tweets
# => Use CORPUS of TWEETS

# --------------------------------------------------------------------------------------------- #
# -------------------------------------------- PREPROCESSING ---------------------------------- #
# --------------------------------------------------------------------------------------------- #
#Set random seed
np.random.seed(24)

#read CSV file containing tweets and labels, using Pandas , to get a dataframe
tweetsData = pd.read_csv('datasets/Sentiment Analysis Dataset.csv', skiprows=[8835, 535881]) #skiping these two rows as they have some bad data
tweetsData.head()

#Dividing the dataset into features and lables
tweets = tweetsData['SentimentText']
labels = tweetsData['Sentiment']

#Lower and split the dialog
#and use regular expression to keep only letters we will use nltk Regular expression package
tkr = RegexpTokenizer('[a-zA-Z@]+')

tweets_split = []

for i, line in enumerate(tweets):
    #print(line)
    tweet = str(line).lower().split()
    tweet = tkr.tokenize(str(tweet))
    tweets_split.append(tweet)

'''
Use pretrained Word2Vec model from google but trim the word list to 50,0000 compared to 300,000 in the original
Google pretrained model
'''

w2vModel = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=50000)

# vectorize a text corpus : Converting Words to integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_split)
X = tokenizer.texts_to_sequences(tweets_split)

# lenght of tweet to consider
maxlentweet = 10
# add padding
X = pad_sequences(X, maxlen=maxlentweet)



# --------------------------------------------------------------------------------------------- #
# ----------------------------------------- EMBEDDING ----------------------------------------- #
# --------------------------------------------------------------------------------------------- #
# Word Embedding is a representation of text where words that have the same meaning have a similar representation.

# TO PUT ALSO BITCOIN TWEETS #
#create a embedding layer using Google pre triained word2vec (50000 words)
embedding_layer = Embedding(input_dim=w2vModel.syn0.shape[0], output_dim=w2vModel.syn0.shape[1], weights=[w2vModel.syn0],
                            input_length=X.shape[1])

# --------------------------------------------------------------------------------------------- #
# ----------------------------------------- DEEP NETWORK -------------------------------------- #
# --------------------------------------------------------------------------------------------- #
# Deep network takes the sequence of embedding vectors as input and converts them to a compressed representation.
# The compressed representation effectively captures all the information in the sequence of words in the text
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout

lstm_out = 80

model = Sequential()
# Add Embedding
model.add(embedding_layer)
model.add(LSTM(units=lstm_out))
# adding softmax for classification
model.add(Dense(1, activation='sigmoid'))
# Try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



#split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size= 0.1, random_state = 24)

#fit model
batch_size = 32
model.fit(X_train, Y_train, epochs=1, verbose=1, batch_size=batch_size)



# model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
#model.fit(X_train, Y_train, batch_size = 128, epochs=25, validation_data=(X_test, Y_test), verbose=2)





# ------------------------------------------------- #
import tensorflow as tf

embedding_layer = tf.keras.layers.Embedding(input_dim=w2vModel.syn0.shape[0], output_dim=w2vModel.syn0.shape[1], weights=[w2vModel.syn0],
                            input_length=X.shape[1])

tf.keras.layers
model = tf.keras.Sequential()

# Add Embedding
model.add(embedding_layer)
# Add a LSTM layer with 128 internal units.
model.add(tf.keras.layers.LSTM(80))

# Add a Dense layer with 10 units and softmax activation.
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          batch_size=batch_size,
          epochs=5)



#analyze the results
score, acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size=batch_size)
y_pred = model.predict(X_test)

#ROC AUC curve
rocAuc = roc_auc_score(Y_test, y_pred)

falsePositiveRate, truePositiveRate, _ = roc_curve(Y_test, y_pred)

plt.figure()

plt.plot(falsePositiveRate, truePositiveRate, color='green',
         lw=3, label='ROC curve (area = %0.2f)' % rocAuc)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic of Sentiiment Analysis Model')
plt.legend(loc="lower right")
plt.show()


#Other accuracy metrices
y_pred = (y_pred > 0.5)

#confusion metrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

#F1 Score, Recall and Precision
print(classification_report(Y_test, y_pred, target_names=['Positive', 'Negative']))