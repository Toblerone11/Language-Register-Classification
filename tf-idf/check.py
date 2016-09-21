from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import re


def parser():
        categories = ['rec.autos', 'rec.motorcycles','rec.sport.baseball', 'rec.sport.hockey',
                      'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast']
        return fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

def getCategory(cat):
    sport = ['rec.autos', 'rec.motorcycles','rec.sport.baseball', 'rec.sport.hockey']
    politics = ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast']
    if cat in sport:
        return "Sport"
    elif cat in politics:
        return "Politics"


def countData(count_vect, data):
    return count_vect.fit_transform(data)

def tfidfData(tfidf, data):
    return tfidf.fit_transform(data)

def classfierNB(data, category):
    return MultinomialNB().fit(data, category)

def classfierSVM(data, category):
    return SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(data, category)

def getText(filename):
    file = open(filename, 'r')
    text = file.read().lower()
    file.close()
    #text = text.lower()
    text = re.sub('[^a-z\ \']+', " ", text)
    return text

def tester(text, classifier, count_vect, tfidf):
    dataCount = count_vect.transform(text)
    dataTFIDF = tfidf.transform(dataCount)
    return classifier.predict(dataTFIDF)

def printResult(predicted, file, docs):
    for doc, category in zip(file, predicted):
        print "Text:"
        print doc
        print "Result:"
        print getCategory(docs.target_names[category])

def main():
    docs = parser()
    count_vect = CountVectorizer()
    tfidf = TfidfTransformer()
    dataCount = countData(count_vect, docs.data)
    dataTFIDF = tfidfData(tfidf, dataCount)
    clf = classfierNB(dataTFIDF, docs.target)
    #SentTest = ['Ron Play with the ball', 'Protection and problems in the middle east']
   # text = 'Ron Play with the ball'
    file = [getText('file.txt')]
    predicted = tester(file, clf, count_vect, tfidf)
    printResult(predicted, file, docs)

main()