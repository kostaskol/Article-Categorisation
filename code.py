#!/bin/python3
import csv
from sys import argv
import numpy as np
from wordcloud import WordCloud
from stop_words import get_stop_words
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from Document import Document
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_distances
stop_word = get_stop_words('en')


def bayes(start=0, end=-1):
    doc = Document('data_sets/train_set.csv', start=start, end=end)

    kf = KFold(n_splits=10)
    fold = 0
    accu = 0
    prec = 0
    rec = 0
    fmes = 0
    avg_auc = 0

    # Vectorize our dataset
    cv = TfidfVectorizer(max_features=200, stop_words=stop_word)
    svd = TruncatedSVD(n_components=5)
    clf = BernoulliNB()
    pipeline = Pipeline([
        ('tf', cv),
        ('svd', svd),
        ('clf', clf)
    ])
    n_classes = 5

    mean_auc = [0.0] * n_classes
    mean_precision = [0] * 100
    mean_recall = np.linspace(0, 1, 100)
    AUCS = []
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    for (train_index, test_index), color in zip(kf.split(doc.data), colors):

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in range(train_index[0], train_index[0] + len(train_index)):
            x_train.append(doc.data[i])
            y_train.append(doc.target[i])
        for i in range(test_index[0], test_index[0] + len(test_index)):
            x_test.append(doc.data[i])
            y_test.append(doc.target[i])

        y_train2 = label_binarize(y_train, classes=[0, 1, 2, 3, 4])

	# Fit, Transform and fit (by the classifier) the dataset
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)

        fold += 1
        accu += accuracy_score(predicted, y_test)
        p, r, f, s = precision_recall_fscore_support(predicted, y_test, average='macro')
        prec += p
        rec += r
        fmes += f

        x_train2 = cv.fit_transform(x_train)
        x_test2 = cv.fit_transform(x_test)
        ovr = OneVsRestClassifier(clf)
        ovr.fit(x_train2.toarray(), y_train2)
        y_score = ovr.predict_proba(x_test2.toarray())

        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:], y_score[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
            mean_auc[i] += roc_auc[i]
            mean_precision[i] += interp(mean_recall, fpr[i], tpr[i])
            mean_precision[i][0] = 0.0
            AUCS.append(roc_auc[i])

    plt.figure()
    for i in range(n_classes):
        mean_precision[i] /= 10
        mean_auc_pr = auc(mean_recall, mean_precision[i])
        plt.plot(mean_recall, mean_precision[i],
                 label='Class ' + doc.target_names[i] + ': Mean AUC = %0.2f' % mean_auc_pr, lw=2)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall')
    plt.legend(loc='lower right')

    plt.show()

    avg_auc = sum(AUCS) / len(AUCS)

    with open(CSV, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['Naive Bayes', accu / fold, prec / fold, rec / fold, fmes / fold, avg_auc / fold])


def svm_clf(start=0, end=-1, prod=False):
    doc = Document('data_sets/train_set.csv', start=start, end=end)

    if not prod:
        kf = KFold(n_splits=10)
        fold = 0
        accu = 0
        prec = 0
        rec = 0
        fmes = 0
        avg_auc = 0

        # Vectorize our dataset
        cv = TfidfVectorizer(max_features=200, stop_words=stop_word)
        svd = TruncatedSVD(n_components=5)
        clf = svm.SVC(kernel='linear', probability=True, random_state=40)
        pipeline = Pipeline([
            ('tf', cv),
            ('svd', svd),
            ('clf', clf)
        ])

        n_classes = 5
        mean_auc = [0.0] * n_classes
        mean_precision = [0] * 100
        mean_recall = np.linspace(0, 1, 100)
        AUCS = []
        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        for (train_index, test_index), color in zip(kf.split(doc.data), colors):

            x_train = []
            y_train = []
            x_test = []
            y_test = []
            for i in range(train_index[0], train_index[0] + len(train_index)):
                x_train.append(doc.data[i])
                y_train.append(doc.target[i])
            for i in range(test_index[0], test_index[0] + len(test_index)):
                x_test.append(doc.data[i])
                y_test.append(doc.target[i])

            pipeline.fit(x_train, y_train)
            predicted = pipeline.predict(x_test)

            fold += 1
            accu += accuracy_score(predicted, np.array(doc.target)[test_index])
            p, r, f, s = precision_recall_fscore_support(predicted, np.array(doc.target)[test_index], average='macro')
            prec += p
            rec += r
            fmes += f

            y_train2 = label_binarize(y_train, classes=[0, 1, 2, 3, 4])
            x_train2 = cv.fit_transform(x_train)
            x_test2 = cv.fit_transform(x_test)
            ovr = OneVsRestClassifier(clf)
            ovr.fit(x_train2.toarray(), y_train2)
            y_score = ovr.predict_proba(x_test2.toarray())

            fpr = {}
            tpr = {}
            roc_auc = {}

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:], y_score[:, i], pos_label=i)
                roc_auc[i] = auc(fpr[i], tpr[i])
                mean_auc[i] += roc_auc[i]
                mean_precision[i] += interp(mean_recall, fpr[i], tpr[i])
                mean_precision[i][0] = 0.0
                AUCS.append(roc_auc[i])

        plt.figure()
        for i in range(n_classes):
            mean_precision[i] /= 10
            mean_auc_pr = auc(mean_recall, mean_precision[i])
            plt.plot(mean_recall, mean_precision[i],
                     label='Class ' + doc.target_names[i] + ': Mean AUC = %0.2f' % mean_auc_pr, lw=2)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall')
        plt.legend(loc='lower right')

        plt.show()

        avg_auc = sum(AUCS) / len(AUCS)

        with open(CSV, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['SVM', accu / fold, prec / fold, rec / fold, fmes / fold, avg_auc / fold])


def word_cloud(tar):
    doc = Document('data_sets/train_set.csv')

    wordcloud = WordCloud().generate(doc.get_clean(tar))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.show()


def k_means(start=0, end=-1):
    # Prepare the csv
    with open('output/clustering_KMeans.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['Cluster#', 'Politics', 'Business', 'Football', 'Film', 'Technology'])

    doc = Document('data_sets/train_set.csv', start=start, end=end)

    vector = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=stop_word, max_features=200)

    x_counts = vector.fit_transform(doc.data)

   

    normalizer = Normalizer(copy=False)
    svd = TruncatedSVD(n_components=199)
    lsa = make_pipeline(svd, normalizer)
    x_counts = lsa.fit_transform(x_counts)

    km = KMeans(5, n_init=1, max_iter=100)
    km.fit(x_counts)
    centroids = km.cluster_centers_

    d = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: []
    }

    for i in range(len(doc.data)):
        dist = cosine_distances(x_counts[i: i+1], centroids)
        minimum = dist[0].tolist().index(min(dist[0].tolist()))
        d[minimum].append(doc.target[i])

    i = 0
    # Create a dictionary for the stats of each centroid
    for key in d.keys():
        stats = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }
	

        for j in d[key]:
	    # Increase the class occurence count by one each time we come across it
            stats[j] += 1

        for j in stats.keys():
	    # Find the percentage of occurence for the specific class
            stats[j] = stats[j] / len(d[key])

        with open('output/clustering_KMeans.csv', 'a') as csvfile:
	    # Write it to the csv
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['Cluster' + str(i + 1), stats[0], stats[1], stats[2], stats[3], stats[4]])
        i += 1


def knn_alg(start=0, end=-1, k=3):

    def get_vote():
        # get each neighbour's class
        # the class of the test document is the
        # same as the most voted one
        d = {}
        for j in neighb:
            t_target = docs.target[j]
            if t_target not in d.keys():
                d[t_target] = 1
            else:
                d[t_target] += 1

        max_votes = 0
        max_target = None
        for key in d.keys():
            if d[key] > max_votes:
                max_votes = d[key]
                max_target = key

        return max_target

    docs = Document('data_sets/train_set.csv', start=start, end=end)
    tfidf = TfidfVectorizer(max_features=200, norm='l2',
                            stop_words=stop_word).fit_transform(docs.data)

    predicted = []

    kf = KFold(n_splits=10)
    accu = 0
    prec = 0
    rec = 0
    fmes = 0
    fold = 0
    for train_index, test_index in kf.split(docs.data):
        for i in range(test_index[0], test_index[0] + len(test_index)):
            # find the cosine similarity using the linear kernel
            cos_sim = linear_kernel(tfidf[i: i+1], tfidf).flatten()

            # find the k nearest neighbours using argsort
            neighb = cos_sim.argsort()[:-k:-1]

            target = get_vote()
            predicted.append(target)

        accu += accuracy_score(predicted, np.array(docs.target)[test_index])
        p, r, f, _ = precision_recall_fscore_support(predicted, np.array(docs.target)[test_index], average='macro')
        prec += p
        rec += r
        fmes += f
        fold += 1
        predicted = []

    with open(CSV, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['K-NN', accu / fold, prec / fold, rec / fold, fmes / fold, 'Unavailable'])

# If prod == True, then we try to predict the test set's 
# documents' classes
def random_forests(start=0, end=-1, prod=False):
    doc = Document('data_sets/train_set.csv', start=start, end=end)
    if not prod:
        kf = KFold(n_splits=10)
        fold = 0
        accu = 0
        prec = 0
        rec = 0
        fmes = 0
        avg_auc = 0

        cv = TfidfVectorizer(max_features=200, stop_words=stop_word)
        svd = TruncatedSVD(n_components=5)
        clf = RandomForestClassifier()
        pipeline = Pipeline([
            ('tf', cv),
            ('svd', svd),
            ('clf', clf)
        ])

        n_classes = 5

        mean_auc = [0.0] * n_classes
        mean_precision = [0] * 100
        mean_recall = np.linspace(0, 1, 100)
        AUCS = []
        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        for (train_index, test_index), color in zip(kf.split(doc.data), colors):

            x_train = []
            y_train = []
            x_test = []
            y_test = []
            for i in range(train_index[0], train_index[0] + len(train_index)):
                x_train.append(doc.data[i])
                y_train.append(doc.target[i])
            for i in range(test_index[0], test_index[0] + len(test_index)):
                x_test.append(doc.data[i])
                y_test.append(doc.target[i])

            y_train2 = label_binarize(y_train, classes=[0, 1, 2, 3, 4])

            pipeline.fit(x_train, y_train)
            predicted = pipeline.predict(x_test)

            fold += 1
            accu += accuracy_score(predicted, y_test)
            p, r, f, s = precision_recall_fscore_support(predicted, y_test, average='macro')
            prec += p
            rec += r
            fmes += f

            x_train2 = cv.fit_transform(x_train)
            x_test2 = cv.fit_transform(x_test)
            ovr = OneVsRestClassifier(clf)
            ovr.fit(x_train2.toarray(), y_train2)
            y_score = ovr.predict_proba(x_test2.toarray())

            fpr = {}
            tpr = {}
            roc_auc = {}

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:], y_score[:, i], pos_label=i)
                roc_auc[i] = auc(fpr[i], tpr[i])
                mean_auc[i] += roc_auc[i]
                mean_precision[i] += interp(mean_recall, fpr[i], tpr[i])
                mean_precision[i][0] = 0.0
                AUCS.append(roc_auc[i])

        plt.figure()
        for i in range(n_classes):
            mean_precision[i] /= 10
            mean_auc_pr = auc(mean_recall, mean_precision[i])
            plt.plot(mean_recall, mean_precision[i],
                     label='Class ' + doc.target_names[i] + ': Mean AUC = %0.2f' % mean_auc_pr, lw=2)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall')
        plt.legend(loc='lower right')
        plt.show()

        avg_auc = sum(AUCS) / len(AUCS)

        with open(CSV, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['Random Forests', accu / fold, prec / fold, rec / fold, fmes / fold, avg_auc / fold])

    else:
        test_doc = Document('data_sets/test_set.csv', test=True)
        cv = TfidfVectorizer(max_features=200, stop_words=stop_word)
        clf = RandomForestClassifier()
        x_counts = cv.fit_transform(doc.data)

        test_counts = cv.transform(test_doc.data)

        clf.fit(x_counts, doc.target)
        predicted = clf.predict(test_counts)

        with open('output/testSet_categories.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['ID', 'Predicted_Category'])
            for doc_id, target in zip(test_doc.id, predicted):
                writer.writerow([doc_id, test_doc.target_names[target]])

CSV = "output/EvaluationMetric_10fold.csv.rm"

if len(argv) < 2:
   print("Need at least 1 argument. Type --help for help")
   exit()
for i in range(1, len(argv)):
    c = argv[i]
    if c == 'w':
        for w in range(5):
            word_cloud(w)
    elif c == 'k':
        k_means()
    elif c == 'rf':
        random_forests()
    elif c == 'b':
        bayes()
    elif c == 'svm':
        svm_clf()
    elif c == 'knn':
        knn_alg(k=20)
    elif c == 'prod':
        random_forests(prod=True)
    elif c == '--help':
        print("Help for ", argv[0])
        print("Options:")
        print("w: word cloud image generation")
        print("k: K means clustering")
        print("rf: Random Forest")
        print("b: Naive Bayes")
        print("svm: SVM")
        print("knn: K-NN")
        print("prod: Predict the class of the documents in data_sets/test_set.csv\n\tusing the Random Forest algorithm")
