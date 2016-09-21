import csv
import sys, os
import cPickle as pickle
import re
import numpy as np
from sklearn.datasets.base import Bunch

CLEAN_CORPUS = "clean_corpus"

DIGIT_REPR = "<!DIGIT!>"
BEGIN_S = re.compile("\s*<s>\s*")
END_S = re.compile("\s*</s>\s*")
START_D = re.compile("\s*<text id=\w+>\s*")
END_D = re.compile("\s*</text>\s*")
STRIP = re.compile("['.,:;()+\s\"]+")
DIGIT = re.compile("[,.\d+]")
GOLD_STD_PAIR_PATT = re.compile("(\w+)\s+(\w+)\s+([A-Z])\s+(\d+(?:\.\d+)?)\s*")

SMOOTH_FACTOR = 2


def createWikiSentenceBased(path_to_corpus):
    with open(path_to_corpus) as raw_c:
        dir = os.path.abspath(
            os.path.join(path_to_corpus, os.pardir))  # TODO check that this is the parent directory of the file.
        print(dir)
        lineNum = 0
        printLine = 0
        with open(dir + os.sep + CLEAN_CORPUS, 'w+') as clean_c:
            sentence = ""
            line = raw_c.readline()
            while line != "":
                line = raw_c.readline()

                lineNum += 1
                printLine += 1
                if printLine == 1000000:
                    print(lineNum)
                    printLine = 0

                if BEGIN_S.match(line):
                    sentence = ""
                elif END_S.match(line):
                    clean_c.write(sentence + "\n")
                elif START_D.match(line):
                    continue
                elif END_D.match(line):
                    continue
                else:
                    clean_word = STRIP.sub("", line).lower()
                    if DIGIT.match(clean_word):
                        clean_word = DIGIT_REPR

                    add = " " if len(sentence) > 0 else ""
                    sentence += add + clean_word


def createWikiDast(path_to_corpus):
    lineNum = 0
    printLine = 0
    data = []
    target = []
    with open(path_to_corpus) as raw_c:
        dir = os.path.abspath(
            os.path.join(path_to_corpus, os.pardir))  # TODO check that this is the parent directory of the file.
        print(dir)
        line = 'first'
        while line != "":
            line = raw_c.readline()

            lineNum += 1
            printLine += 1
            if printLine == 1000000:
                print(lineNum)
                printLine = 0
            if lineNum == 364270:
                break
            data.append(line)
            target.append(1)
    dast = Bunch()
    dast.data = data
    dast.target = target
    # dast.target = numpy.zeros(shape=(lineNum), dtype='int32')
    # docs = {'data': data, 'target':np.asarray(target)}
    return dast


def createTwitterDast(path):
    csv.field_size_limit(sys.maxsize)
    data = []
    target = []
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            sent = ""
            for counter, word in enumerate(row):
                if counter > 5 and counter + 1 < len(row):
                    sent = sent + " " + word
            data.append(sent)
            target.append(0)
            # print sent
    docs = Bunch()
    docs.data = data
    docs.target = target
    return docs


def createPWKP(path):
    sentPack = []
    data = []
    target = []
    with open(path) as f:
        line = f.readline()
        while line != "":
            while line != "\n":
                sentPack.append(line)
                line = f.readline()
            if len(sentPack) != 0:
                data.append(sentPack[0])
                target.append(0)
                data.append(sentPack[1])
                target.append(1)
            sentPack = []
            line = f.readline()
    docs = Bunch()
    docs.data = data
    docs.target = target
    return docs


def createESWiki(enPath, simPath):
    data = []
    target = []
    with open(enPath) as fsim:
        line = fsim.readline()
        while line != "":
            data.append(line)
            target.append(0)
            line = fsim.readline()

    with open(simPath) as fen:
        line = fen.readline()
        while line != "":
            data.append(line)
            target.append(1)
            line = fen.readline()
    return data, target

def createFullCategoryESWiki(enPathList, simPathList):
    data = []
    target = []
    for index, enPath in enumerate(enPathList):
        d, t = createESWiki(enPath, simPathList[index])
        data.extend(d)
        target.extend(t)

    docs = Bunch()
    docs.data = data
    docs.target = target
    return docs

def uniteCorps(obj1, obj2):
    obj1.data = obj1.data + obj2.data
    obj1.target = np.asarray(obj1.target + obj2.target)
    return obj1


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
