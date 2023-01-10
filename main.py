import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Ref:http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from utils import process_tweet, convert_tweets_to_doc_vec, plot_confusion_matrix

############################################# Data Acquisition ######################################

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

x_train = train_pos + train_neg
x_test = test_pos + test_neg

# Create the numpy array of positive labels and negative labels.
# Combine positive and negative labels
y_train = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
y_test = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# print("train_y.shape = " + str(y_train.shape))  # (8000, 1)
# print("test_y.shape = " + str(y_test.shape))  # (2000, 1)
# exit(159)
############################################# Feature Engineering ##################################
####### BOW & TFIDF
vect = TfidfVectorizer(preprocessor=process_tweet)  # instantiate a Vectorizer
X_train_dtm = vect.fit_transform(x_train)  # use it to extract features from training data
X_test_dtm = vect.transform(x_test)  # transform testing data (using training data's features)
# print(X_train_dtm.shape, X_test_dtm.shape)  # (8000, 8648) (2000, 8648)
# print(type(X_train_dtm))
# print(X_train_dtm[0].shape)

####### Word2Vec & Doc2Vec
from gensim.models import Word2Vec

all_tweets_processed = []
for tweet in x_train:
    processed_tweet = process_tweet(tweet, return_list=True)
    if processed_tweet:
        all_tweets_processed.append(processed_tweet)

word2vec = Word2Vec(sentences=all_tweets_processed, vector_size=200, window=7, min_count=1, workers=4)
word2vec.save("word2vec.model")
w2v_model = Word2Vec.load("word2vec.model")
# print(w2v_model.wv.vectors.shape) # (9085, 100) --> (vocab size, vector size)

X_train_using_doc2vec = convert_tweets_to_doc_vec(x_train, w2v_model)
X_test_using_doc2vec = convert_tweets_to_doc_vec(x_test, w2v_model)

print(X_train_using_doc2vec.shape)
print(X_test_using_doc2vec.shape)

# vect_tech = "BoW-TfIDF"
vect_tech = "Doc2Vec"

if vect_tech == "Doc2Vec":
    X_train_final = X_train_using_doc2vec
    X_test_final = X_test_using_doc2vec
else:
    X_train_final = X_train_dtm.toarray()
    X_test_final = X_test_dtm.toarray()
############################################# Modeling #############################################
models = [GaussianNB(), LogisticRegression(), SVC(kernel='linear', probability=True)]

for model in models:
    model.fit(X_train_final, y_train)
    y_hat = model.predict(X_test_final)

############################################# Model Evaluation #####################################
    # print the confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_hat)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cnf_matrix, classes=['Negative', 'Positive'], normalize=True,
                          title='Confusion matrix with all features')

    # calculate AUC: Area under the curve(AUC) gives idea about the model efficiency:
    # Further information: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    y_pred_prob = model.predict_proba(X_test_final)[:, 1]

    print("Accuracy: " + str(accuracy_score(y_test, y_hat) * 100))
    print("Precision: " + str(precision_score(y_test, y_hat, labels=model.classes_)))
    print("Recall: " + str(recall_score(y_test, y_hat, labels=model.classes_)))
    print("F1 Score: " + str(f1_score(y_test, y_hat, labels=model.classes_)))
    print("ROC_AOC_Score: ", roc_auc_score(y_test, y_pred_prob))
    print("------------------------------------------------------------")
