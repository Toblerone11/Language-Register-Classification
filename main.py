import os
from pprint import pprint

from sklearn.metrics import accuracy_score

from sgdModel import sgdModel
from cbowModel import cbowModel
from datatools import get_samples_iterator
from sentences_repr import SentenceTool

BOTH_PATH = r".\PWKP_108016\PWKP_108016"
EN_PATH = r".\PWKP_108016\en_wiki.sentences"
SIMPLE_PATH = r"./PWKP_108016/simple_wiki.sentences"
TRAIN_PATH = r".\PWKP_108016\train_set.sentences"
LABEL_PATH = r".\corpora\PWKP_108016\labels.lbl"

SIMPLE_TEST_PATH = r"./PWKP_108016/simple_test.sentences"
EN_TEST_PATH = r"./PWKP_108016/en_test.sentences"

EN_MODEL_PATH = "./en_model.model"
SIMPLE_MODEL_PATH = "./simple_model.model"

CLF_PATH = r".\sgdClassifier.pkl"
MODEL_PATH = r".\cbowClass"
MEM_LIMIT = 25000

SIMPLE_LABEL = 1
EN_LABEL = 0
CLASSES = [EN_LABEL, SIMPLE_LABEL]


def evaluate_model(y_bar, y):
    TP, TN, FP, FN = 0, 0, 0, 0
    loss = 0
    sum_test = len(y)
    
    for i in range(len(y)):
        if y_bar[i] == 0:
            if y[i] == 0:
                TP += 1
            else:
                FP += 1
                loss += 1
            
        else:
            if y[i] == 1:
                TN += 1
            else:
                FN += 1
                loss += 1
    
    print("TP: {TP}\tFP: {FP}\tFN: {FN}\tTN: {TN}".format(TP=TP, FP=FP, FN=FN, TN=TN))
    
    recall = float(TP) / (TP + FN) # amount we cover compare to all what we should
    precision = float(TP) / (TP + FP) # amount we corrected as True compare to all what is really True.
    F1 = 2 * (recall * precision) / (recall + precision)
    
    print("recall: ", recall)
    print("precision: ", precision)
    print("F1: ", F1)
    print("Loss: ", float(loss) / sum_test)

def get_next_label(path_to_labels):
    with open(path_to_labels, 'r') as labelf:
        line = labelf.readline()
        while(line != ""):
            label = int(line[:-1])
            yield label

            line = labelf.readline()

if __name__ == "__main__":
    
    # train SGD
    # sgd = sgdModel()
    # sgd.fit_large_scale(get_samples_iterator(category='parallel', register='both'), get_next_label(LABEL_PATH), classes=CLASSES)
    # y_bar, y = sgd.predict_large_scale(get_samples_iterator(category='parallel', register='both_test', include_tag=True))
    # score = accuracy_score(y_bar, y)
    # print(score)
    # F1_score = evaluate_model(result, Y_test)
    # print(score)
    
    # train CBOW
    cbow = cbowModel(MODEL_PATH)
    
    # train on categories separetly   
    for cat in ['parallel', 'arts', 'sports', 'tech']:
        print("Train on ", cat)
        cbow.fit_large_scale(cat, 'en', cat, 'simple', classes=CLASSES)
        print("Test on ", cat)
        y_bar, y = cbow.predict_large_scale('parallel', 'en_test', 'parallel', 'simple_test', classes=CLASSES)
        score = accuracy_score(y_bar, y)
        print(score)
        F1_score = evaluate_model(y_bar, y)
        print(score)
    
        

"""

    try:
        assert(len(result) == len(Y_test))
    except:
        print(len(result))
        print(len(Y_test))
        raise
    score = accuracy_score(result, Y_test)
    F1_score = evaluate_model(result, Y_test)
    print(score)
    
    
"""
    
    
            

    
    