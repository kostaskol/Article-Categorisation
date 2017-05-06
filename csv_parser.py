from pandas import read_csv
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

stop_word = get_stop_words('en')




def bayes(start=0, end=-1):
    doc = Document('train_set.csv', start=start, end=end)

    count_vect = TfidfVectorizer(max_features=1000)
    count_vect.fit(doc.data)
    X_train_counts = count_vect.transform(doc.data)

    clf = MultinomialNB().fit(X_train_counts, doc.target)

    docs_new = Document('train_set.csv', start=5001)
    X_new_counts = count_vect.transform(docs_new.data)

    predicted = clf.predict(X_new_counts)

    for docs, category in zip(docs_new.data, predicted):
        print('%r => %s' % (docs, doc.target_names[category]))


def svm_clf(start=0, end=-1):
    doc = Document('train_set.csv', start=start, end=end)

    cv = TfidfVectorizer(max_features=1000) 
    cv.fit(doc.data)
    x_train_counts = cv.transform(doc.data)

    clf = svm.LinearSVC()
    clf.fit(x_train_counts, doc.target)

    docs_new = Document('train_set.csv', start=5001)
    x_new_counts = cv.transform(docs_new.data)

    predicted = clf.predict(x_new_counts)

    for docs, category in zip(docs_new.data, predicted):
        print('%r => %s' % (docs, doc.target_names[category]))


def word_cloud(tar):
    doc = Document('train_set.csv')

    wordcloud = WordCloud().generate(doc.get_clean(tar))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.show()


def k_means(start=0, end=-1):
    doc = Document('train_set.csv', start=start, end=end)

    vector = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, stop_words=stop_word)

    x_counts = vector.fit_transform(doc.data)

    centroids, labels, inertia = cluster.k_means(x_counts, n_clusters=5)

    print("Cent: ", centroids, ", labels ", labels, ', inertia ', inertia)

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
''' Under construction '''
def knn_alg(start=0, end=-1, k=3):
    nltk.download('punkt')

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

    docs = Document('train_set.csv', start=start, end=end)
    tfidf = TfidfVectorizer().fit_transform(docs.data)
    from sklearn.metrics.pairwise import linear_kernel
    for i in range(len(docs.data)):
        cos_sim = linear_kernel(tfidf[i:i+1], tfidf).flatten()

        neighb = cos_sim.argsort()[:-k:-1]

        target = get_vote(docs, neighb)
        print("Document: ", docs.data[i])
        print("Target ---------------> ", docs.target_names[target])

    #for doc in docs.data:
    return
    doc = docs.data[1]
    neighbours = []
    for i in range(2, len(docs.data)):
            neighbours.append((cosine_sim(doc, docs.data[i]), i))

    n_neighbours = []
    for i in neighbours:
        if i[0] != 0:
            n_neighbours.append(i)
    # get the max value from the list
    min_distances = []
    
    for l in range(k):
        index = neighbours.index(min(neighbours, key=lambda t: t[0]))
        
        min_distances.append(neighbours[index][1])
        del neighbours[index]
    
    n_neighbours.sort()
    for i in range(k):
        print("Closest neighbours are: ", n_neighbours[i])

   # print("Min distances are: ", min_distances)
    print("Prediction for document ", doc)
    print("------> ", docs.target_names[get_vote(docs, n_neighbours[:k])])


def random_forests(start=0, end=-1):
    doc = Document(csv_file='train_set.csv', start=start, end=end)
    cv = TfidfVectorizer(max_features=100)
    cv.fit(doc.data)
    count = cv.transform(doc.data)

    clf = RandomForestClassifier(n_estimators=5)
    clf = clf.fit(count, doc.target)

    docs_new = Document('train_set', start=5001)
    count_new = cv.transform(docs_new.data)

    predicted = clf.predict(count_new)

    for docs, category in zip(docs_new.data, predicted):
        print("%r => %s" % (docs, doc.target_names[category]))


for i in range(5):
    word_cloud(i)
# knn_alg(k=20)
# random_forests(5000)
# bayes(end=10000)
# svm_clf(10000)
# random_forests(end=10000)
# k_means(end=10000)

