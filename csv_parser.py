from pandas import read_csv
import numpy
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import re
from stop_words import get_stop_words
from sklearn import svm

stop_word = get_stop_words('en')


class Document:
    def __init__(self, csv_file, head):
        self.csv = csv_file
        self.target_names = {
            0: 'Politics',
            1: 'Film',
            2: 'Football',
            3: 'Business',
            4: 'Technology'
        }

        self.data = []
        self.target = []

        df = read_csv('train_set.csv', sep='\t')

        A = numpy.array(df.head(head))

        for i in range(A.shape[0]):
            t = str(A[i][4])
            self.data.append(str(A[i][3]))
            if t == 'Politics':
                self.target.append(0)
            elif t == 'Film':
                self.target.append(1)
            elif t == 'Football':
                self.target.append(2)
            elif t == 'Business':
                self.target.append(3)
            elif t == 'Technology':
                self.target.append(4)

    def get_text(self, target):
        string = ""
        for i in range(len(self.data)):
            if self.target[i] == target:
                string += self.data[i]

        return string

    def get_clean(self, target):
        string = self.get_text(target)
        global stop_word
        for i in stop_word:
            tmp = re.compile(re.escape(' ' + i + ' '), re.IGNORECASE)
            string = tmp.sub(' ', string)

        return string


def bayes(head):
    doc = Document('train_set.csv', head)

    count_vect = CountVectorizer(stop_words=stop_word)
    count_vect.fit(doc.data)
    X_train_counts = count_vect.transform(doc.data)

    clf = MultinomialNB().fit(X_train_counts, doc.target)

    docs_new = ['We are programming in python', 'Money']
    X_new_counts = count_vect.transform(docs_new)

    predicted = clf.predict(X_new_counts)

    for docs, category in zip(docs_new, predicted):
        print('%r => %s' % (docs, doc.target_names[category]))


def svm_clf(head):
    doc = Document('train_set.csv', head)

    cv = CountVectorizer(stop_words=stop_word)
    cv.fit(doc.data)
    x_train_counts = cv.transform(doc.data)

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x_train_counts, doc.target)

    dec = clf.decision_function([[1]])
    print(dec.shape)

    docs_new = ['We are programming in python', 'Money']
    x_new_counts = cv.transform(docs_new)

    predicted = clf.predict(x_new_counts)

    for docs, category in zip(docs_new, predicted):
        print('%r => %s' % (docs, doc.target_names[category]))


def word_cloud(tar):
    doc = Document('train_set.csv')

    wordcloud = WordCloud().generate(doc.get_clean(tar))

    import matplotlib.pyplot as plt
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.show()


def k_means(hear):
    pass

svm_clf(5000)
