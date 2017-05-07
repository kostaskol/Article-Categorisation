#!/bin/python3

# TODO: Clean this up if possible OR delete this comment
from pandas import read_csv
from sys import argv
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import re
from stop_words import get_stop_words
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import math
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.stem.porter import PorterStemmer
import string as String
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from Document import Document
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import KFold
stop_word = get_stop_words('en')




def bayes(start=0, end=-1):
    doc = Document('train_set.csv', start=start, end=end)
    # Σπάμε το set σε 10 κομμάτια
    kf = KFold(n_splits=10)
    fold = 0
    # Φτιάχνουμε τον CV
    count_vect = TfidfVectorizer(max_features=1000)
    # Του κάνουμε το training
    count_vect.fit(doc.data)
    for train_index, test_index in kf.split(doc.data):
        # Κάνουμε transform από text σε floats (vectors)
        # το training και το test set μας
        X_train = count_vect.transform(np.array(doc.data)[train_index])
        X_test = count_vect.transform(np.array(doc.data)[test_index])

        # Κάνουμε και fit το training set με τον MultNB (Bayes)
        clf = MultinomialNB().fit(X_train, np.array(doc.target)[train_index])
        predicted = clf.predict(X_test)
        fold += 1
        # Εκτυπώνουμε το Accuracy και τα Prec - Recall - F1-score - Support
        print("Fold ", fold)
        print("Accuracy: ", clf.score(X_test, np.array(doc.target)[test_index]))
        print(classification_report(predicted, np.array(doc.target)[test_index], target_names=doc.target_names_str))
        # Πιο κάτω υπάρχει το ROC plot. Δεν μπορώ να βρω πως γίνεται
        # TODO:1. Είτε λύσε το, είτε φτιάξ'το να εμφανίζει έστω αυτό που 
        # μας έδειχνε χθες.
        # TODO:2. Σβήσε τα 2 TODOs για να μην υπάρχουν στην παράδοση
    return
'''
    count_vect.fit(doc.data)
    X_train_counts = count_vect.transform(doc.data)

    clf = MultinomialNB().fit(X_train_counts, doc.target)

    docs_new = Document('train_set.csv', start=5001)
    X_new_counts = count_vect.transform(docs_new.data)

    Prints the precision(?)
    print(clf.score(X_new_counts, docs_new.target))
    return

    predicted = clf.predict(X_new_counts)
    Precision, Recall, F1-Score, Support
    k = classification_report(docs_new.target, predicted)
    '''
    from sklearn.multiclass import OneVsRestClassifier
    clf = OneVsRestClassifier(clf)
    y_score = clf.fit(X_train_counts, doc.target).predict_proba(X_new_counts)
    from sklearn.preprocessing import label_binarize
    y = label_binarize(docs_new.target, classes=[0, 1, 2, 3, 4])
    n_counts = y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_counts):
        fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



def svm_clf(start=0, end=-1):
    doc = Document('train_set.csv', start=start, end=end)

    # Vectorize our dataset
    cv = TfidfVectorizer(max_features=1000) 
    cv.fit(doc.data)
    x_train_counts = cv.transform(doc.data)

    # Create a new SVC classifier with a linear kernel
    # and train it
    clf = svm.LinearSVC()
    clf.fit(x_train_counts, doc.target)

    # Useless from here on out
    # TODO: Re-write this using 10-fold validation (example @bayes)
    docs_new = Document('train_set.csv', start=5001)
    x_new_counts = cv.transform(docs_new.data)

    predicted = clf.predict(x_new_counts)

    for docs, category in zip(docs_new.data, predicted):
        print('%r => %s' % (docs, doc.target_names[category]))


def word_cloud(tar):
    # TODO: Write the library we used (word_cloud) in the README file
    doc = Document('train_set.csv')

    wordcloud = WordCloud().generate(doc.get_clean(tar))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.show()


def k_means(start=0, end=-1):
    #TODO: Figure out how to get percentage of each class in a cluster (do this last)
    doc = Document('train_set.csv', start=start, end=end)

    vector = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, stop_words=stop_word)

    x_counts = vector.fit_transform(doc.data)

    centroids, labels, inertia = cluster.k_means(x_counts, n_clusters=5)


    d = {}
    for label in labels:
        if label in d.keys():
            d[label] += 1
        else:
            d[label] = 1

    
    for key in d.keys():
        print(d[key] / doc.size)
    # doc = Document(csv_file='train_set.csv', end=end)
    # cv = CountVectorizer()
    # cv.fit(doc.data)
    # X = cv.transform(doc.data)
    # Y = doc.target
    # clf = KMeans(n_clusters=5)
    #
    # clf.fit(X)
    #
    # labels = clf.labels_
    # centroids = clf.cluster_centers_
    # for i in range(5):
    #     # select only data observations with cluster label == i
    #     ds = X[np.where(labels == i)]
    #     # plot the data observations
    #     plt.plot(ds[:, 0], ds[:, 1], 'o')
    #     # plot the centroids
    #     lines = plt.plot(centroids[i, 0], centroids[i, 1], 'kx')
    #     # make the centroid x's bigger
    #     plt.setp(lines, ms=15.0)
    #     plt.setp(lines, mew=2.0)
    # plt.show()


from scipy import spatial
def knn_alg(start=0, end=-1, k=3):
    # This will download nltk.puncuation if it doesn't already exist
    nltk.download('punkt')

    # Remove stems, punctuation and stop words
    stemmer = nltk.stem.porter.PorterStemmer()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

    def stem_tokens(tokens):
        return [stemmer.stem(item) for item in tokens]

    def normalize(text):
        return stem_tokens(nltk.wordpunct_tokenize(text.lower().translate(remove_punctuation_map)))

    def cosine_sim(text1, text2):
        tfidf = TfidfVectorizer(max_features=500).fit_transform([text1, text2])
        #return (tfidf * tfidf.T).A[0, 1]
        return 1 - spatial.distance.cosine(tfidf, tfidf.T)

    def get_vote(docs, dist):
        # Get each neighbour's class 
        # The class of the test document is the
        # same as the most voted one
        d = {}
        for j in dist:
            target = docs.target[j]
            if target not in d.keys():
                d[target] = 1
            else:
                d[target] += 1

        max_votes = 0
        max_target = None
        for key in d.keys():
            if d[key] > max_votes:
                max_votes = d[key]
                max_target = key

        return max_target

    # TODO: Perform 10-fold validation
    # on this one
    docs = Document('train_set.csv', start=start, end=end)
    tfidf = TfidfVectorizer().fit_transform(docs.data)
    from sklearn.metrics.pairwise import linear_kernel
    for i in range(len(docs.data)):
        # Find the cosine similarity using the linear kernel
        cos_sim = linear_kernel(tfidf[i:i+1], tfidf).flatten()

        # Find the nearest neighbours using argsort (sklearn)
        neighb = cos_sim.argsort()[:-k:-1]

        target = get_vote(docs, neighb)
        print("Document: ", docs.data[i])
        print("Target ---------------> ", docs.target_names[target])

    #for doc in docs.data:
    return
    

def random_forests(start=0, end=-1):
    # Same as above
    doc = Document(csv_file='train_set.csv', start=start, end=end)
    cv = TfidfVectorizer(max_features=100)
    cv.fit(doc.data)
    count = cv.transform(doc.data)

    # n_estimators=5 means that we have 5 classes
    # TODO: Perform 10-fold validation
    clf = RandomForestClassifier(n_estimators=5)
    clf = clf.fit(count, doc.target)

    docs_new = Document('train_set', start=5001)
    count_new = cv.transform(docs_new.data)

    predicted = clf.predict(count_new)

    for docs, category in zip(docs_new.data, predicted):
        print("%r => %s" % (docs, doc.target_names[category]))

c = argv[1]
if c == 'w':
    for i in range(5):
        word_cloud(i)
elif c == 'k':
    knn_alg(k=20)
elif c == 'f':
    random_forests(5000)
elif c == 'b':
    bayes()
elif c == 'sv':
    svm_clf(10000)
elif c == 'km':
    k_means(end=1000)

