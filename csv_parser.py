from pandas import read_csv
import numpy
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
w = 5
word_cloud = [[] for x in range(w)]
cvs = []

'''
cv = CountVectorizer(min_df=0, charset_error="ignore",
                         stop_words="english", max_features=200)

counts = cv.fit_transform([text]).toarray().ravel()
words = np.array(cv.get_feature_names())
# normalize
counts = counts / float(counts.max())
'''
'''
A = numpy.array(df.head(1000))

for i in range(A.shape[0]):
    t = str(A[i][4])
    if t == 'Politics':
        if len(word_cloud[0]) == 0:
            word_cloud[0].append('Politics')
        word_cloud[0].append(str(A[i][3]))
    elif t == 'Film':
        if len(word_cloud[1]) == 0:
            word_cloud[1].append('Film')
        word_cloud[1].append(str(A[i][3]))
    elif t == 'Football':
        if len(word_cloud[2]) == 0:
            word_cloud[2].append('Football')
        word_cloud[2].append(str(A[i][3]))
    elif t == 'Business':
        if len(word_cloud[3]) == 0:
            word_cloud[3].append('Business')
        word_cloud[3].append(str(A[i][3]))
    elif t == 'Technology':
        if len(word_cloud[4]) == 0:
            word_cloud[4].append('Technology')
        word_cloud[4].append(str(A[i][3]))
        '''


class Document:
    def __init__(self, csv_file):
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

        A = numpy.array(df.head(5000))

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


doc = Document('train_set.csv')

count_vect = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
count_vect.fit(doc.data)
X_train_counts = count_vect.transform(doc.data)

clf = MultinomialNB().fit(X_train_counts, doc.target)

docs_new = ['We are programming in python', 'Money']
X_new_counts = count_vect.transform(docs_new)

predicted = clf.predict(X_new_counts)

for docs, category in zip(docs_new, predicted):
    print('%r => %s' % (docs, doc.target_names[category]))