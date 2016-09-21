import corpusTools as c
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets.base import Bunch



class TfidfModel:

    def __init__(self):
        #self.__dictCorpus = createDict(corpus1, corpus2)
        #tw = load_obj("twitter")
        #tw = c.createTwitterDast('1.csv')
        #wiki = c.createWikiDast('clean_corpus')
        #self.__dictCorpus = c.uniteCorps(tw, wiki)
        #self.__dictCorpus = c.createPWKP("pwkp")
        #enSen = ["Sports_en.sentences", "Technology_en.sentences", "Arts_en.sentences"]
        #simpleSen = ["Sports_simple.sentences", "Technology_simple.sentences", "Arts_simple.sentences"]
        enSen = ["Arts_en.sentences"]
        simpleSen =["Arts_simple.sentences"]
        enSen = ["Technology_en.sentences"]
        simpleSen =["Technology_simple.sentences"]
        #enSen = ["Sports_en.sentences"]
        #simpleSen =["Sports_simple.sentences"]
        self.__dictCorpus = c.createFullCategoryESWiki(enSen, simpleSen)
        self.__countVec = CountVectorizer(decode_error='ignore')
        self.__tfidifVec = TfidfTransformer()

    def __preProcessData(self):
        return train_test_split(self.__dictCorpus.data, self.__dictCorpus.target, test_size=0.3, random_state=42)

    def __processTrained(self, data):
        countedData = self.__countVec.fit_transform(data)
        tfidfData = self.__tfidifVec.fit_transform(countedData)

        return tfidfData

    def __processTest(self, data):
        countedData = self.__countVec.transform(data)
        tfidfData = self.__tfidifVec.transform(countedData)

        return tfidfData

    def __processData(self):
        X_train, X_test, y_train, y_test = self.__preProcessData()
        self.__tfidfTrainData = Bunch()
        self.__tfidfTrainData.data = X_train
        self.__tfidfTrainData.target = np.asarray(y_train)
        self.__tfidfTrainData.processed = self.__processTrained(X_train)
        #self.__tfidfTrainData = self.__processTrained(X_train)
        self.__tfidfTestData = Bunch()
        self.__tfidfTestData.data = X_test
        self.__tfidfTestData.target = np.asarray(y_test)
        self.__tfidfTestData.processed = self.__processTest(X_test)
        #self.__tfidfTestData = self.__processTest(X_test)
        #self.__tfidfTrainTarget = y_train
        #self.__tfidifTestTarget = y_test

    def getTestData(self):
        return self.__tfidfTestData.processed
    def getTestTarget(self):
        return self.__tfidfTestData.target

    def createClassifier(self):
        self.__processData()
        return MultinomialNB().fit(self.__tfidfTrainData.processed, self.__tfidfTrainData.target)


def validation(classifier, data, target):
    result = clf.predict(data)
    wikiCounter = 0
    twtCounter = 0
    wikiPredictCounter = 0
    twtPredicCounter = 0
    for index, r in enumerate(result):
        if r == 0 and target[index] == 0:
            twtCounter += 1
        elif r == 1 and target[index] == 1:
            wikiCounter += 1
    wikiPossible = sum(target)
    twtPossible = len(target) - wikiPossible
    # for r in result:
    #     if r == 1:
    #         twtPredicCounter += 1
    #     elif r == 2:
    #         wikiPredictCounter += 1

    print "Twitter: " + str(float(twtCounter) / twtPossible)
    print "Wiki " + str(float(wikiCounter) / wikiPossible)
    print "Score " + str(clf.score(data, target))

def convertListToUnicode(dlist):
    newList = []
    for elem in dlist:
        newList.append(unicode(elem))
    return newList


g = TfidfModel()
clf = g.createClassifier()
validation(clf, g.getTestData(), g.getTestTarget() )
