import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import twitter_samples
from sklearn.model_selection import train_test_split

from utils import build_freqs, process_tweet

############################################# Data Acquisition
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

## Splitting -- Wrong splitting
# x = all_negative_tweets + all_positive_tweets
# y = np.append(np.ones(shape=(len(all_positive_tweets), 1)), np.zeros(shape=(len(all_negative_tweets), 1)), axis=0)
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# Create the numpy array of positive labels and negative labels.
# Combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

# create frequency dictionary
freqs = build_freqs(train_x, train_y)


# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))


############################################# Text Cleaning and PreProcessing
## in utils.py -> process_tweet()
############################################# Feature Engineering
def extract_features(processed_tweet, freqs):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    # loop through each word in the list of words
    for word in processed_tweet:
        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1.0), 0)
        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0)
    return x


# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i, tweet in enumerate(train_x):
    processed_tweet = process_tweet(tweet)
    X[i, :] = extract_features(processed_tweet, freqs)
Y = train_y

############################################# Modeling
from sklearn.linear_model import LogisticRegression

logisticClassifier = LogisticRegression(random_state=42).fit(X, np.ravel(Y))

from sklearn.svm import SVC

svc = SVC(random_state=42, probability=True).fit(X, np.ravel(Y))

from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()
naive_bayes.fit(X, np.ravel(Y))

############################################# Evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

models = [logisticClassifier, svc, naive_bayes]
tweet_features = [extract_features(process_tweet(tweet), freqs) for tweet in test_x]
print("-------- Evaluating --------")
for model in models:
    y_hats = []
    for tweet in tweet_features:
        pred = model.predict(tweet)
        y_hats.append(pred)
    print(str(model))
    print("Accuracy: " + str(accuracy_score(test_y, y_hats) * 100))
    print("Precision: " + str(precision_score(test_y, y_hats, labels=model.classes_)))
    print("Recall: " + str(recall_score(test_y, y_hats, labels=model.classes_)))
    print("F1 Score: " + str(f1_score(test_y, y_hats, labels=model.classes_)))
    print("---------------------------------------")
    cm = confusion_matrix(test_y, y_hats, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()

my_tweet = 'I love good'
processed_tweet = process_tweet(my_tweet)
tweeta = extract_features(processed_tweet, freqs)
print(logisticClassifier.predict(tweeta))
print(logisticClassifier.predict_proba(tweeta)[:, 0])
print(logisticClassifier.predict_proba(tweeta)[:, 1])

print("---------------------------------------")

print(svc.predict(tweeta))
print(svc.predict_proba(tweeta)[:, 0])
print(svc.predict_proba(tweeta)[:, 1])

print("---------------------------------------")

print(naive_bayes.predict(tweeta))
print(naive_bayes.predict_proba(tweeta)[:, 0])
print(naive_bayes.predict_proba(tweeta)[:, 1])
