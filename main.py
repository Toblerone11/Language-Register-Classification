import os
from words_repr import init_word2vec, get_model, get_sentences_iter, line_size
from sentences_repr import SentenceTrainer
from sklearn.linear_model import SGDClassifier
import numpy as np
from pprint import pprint
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from gensim.matutils import Dense2Corpus

BOTH_PATH = r"./PWKP_108016/PWKP_108016"
EN_PATH = r"./PWKP_108016/en_wiki.sentences"
SIMPLE_PATH = r"./PWKP_108016/simple_wiki.sentences"
TRAIN_PATH = r"./PWKP_108016/train_set.sentences"
LABEL_PATH = r"./PWKP_108016/labels.lbl"

SIMPLE_TEST_PATH = r"./PWKP_108016/simple_test.sentences"
EN_TEST_PATH = r"./PWKP_108016/en_test.sentences"

EN_MODEL_PATH = "./en_model.model"
SIMPLE_MODEL_PATH = "./simple_model.model"

CLF_PATH = "./sgdClassifier.pkl"
MEM_LIMIT = 25000

SIMPLE_LABEL = 1
EN_LABEL = 0


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
    forcetrain=True
    print("training en model")
    init_word2vec(EN_PATH, EN_MODEL_PATH, forcetrain=forcetrain)
    words_model = get_model()
    sent_trainer = SentenceTrainer(words_model)

    print("get scores")
    en_test_iter = get_sentences_iter(EN_TEST_PATH)
    simple_test_iter = get_sentences_iter(SIMPLE_TEST_PATH)
    en_scores_on_en_model = sent_trainer.total_score(en_test_iter, 22000)
    simple_scores_on_en_model = sent_trainer.total_score(simple_test_iter, 23000)

    print("training simple model")
    init_word2vec(SIMPLE_PATH, SIMPLE_MODEL_PATH, forcetrain=forcetrain)
    words_model = get_model()
    sent_trainer = SentenceTrainer(words_model)

    print("get scores")
    en_test_iter = get_sentences_iter(EN_TEST_PATH)
    simple_test_iter = get_sentences_iter(SIMPLE_TEST_PATH)
    en_scores_on_simple_model = sent_trainer.total_score(en_test_iter, 22000)
    simple_scores_on_simple_model = sent_trainer.total_score(simple_test_iter, 23000)

    print("evaluation")
    y_bar_simple = [None for _ in range(len(simple_scores_on_simple_model))]
    y_simple = [SIMPLE_LABEL for _ in range(len(simple_scores_on_simple_model))]
    for i in range(len(simple_scores_on_simple_model)):
        simple_liklihood = simple_scores_on_simple_model[i]
        en_liklihood = simple_scores_on_en_model[i]
        if simple_liklihood < en_liklihood:
            label = EN_LABEL
        else:
            label = SIMPLE_LABEL

        y_bar_simple[i] = label

    y_bar_en = [None for _ in range(len(en_scores_on_en_model))]
    y_en = [EN_LABEL for _ in range(len(en_scores_on_en_model))]
    for i in range(len(en_scores_on_en_model)):
        simple_liklihood = en_scores_on_simple_model[i]
        en_liklihood = en_scores_on_en_model[i]
        if en_liklihood < simple_liklihood:
            label = SIMPLE_LABEL
        else:
            label = EN_LABEL

        y_bar_en[i] = label

    y_bar = y_bar_simple + y_bar_en
    y = y_simple + y_en
    score = accuracy_score(y_bar, y)
    print(score)
        






"""
    # building classifier
    forcetrain = False
    if not os.path.exists(CLF_PATH) or forcetrain:
        sgd = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.001, fit_intercept=False)
        classes = [0, 1]
        X_train = np.array([])
        y_train = np.array([])
        train_iter = get_sentences_iter(TRAIN_PATH)
        labels_iter = get_next_label(LABEL_PATH)

        # building train samples
        print("Training model")    
        for sentence in train_iter:
            ngrams_vectors = sent_trainer.to_vector(sentence)
            np.append(X_train, ngrams_vectors)
            label = next(labels_iter)
            np.append(y_train, np.array([label for _ in range(len(ngrams_vectors))]))
            if len(X_train) > MEM_LIMIT:
                X_train = np.array(X_train)
                sgd.partial_fit(X_train, y_train, classes=classes)
                X_train = []
                y_train = []

""""""
        ### Trying to put the matrix in gensim vector space ###
        model = Dense2Corpus(X_train)
        print(dir(model))
""""""
        
        joblib.dump(sgd, CLF_PATH)
        
    sgd = joblib.load(CLF_PATH) 
    
    # bulding test data
    en_sent_iter = get_sentences_iter(EN_TEST_PATH)
    simple_sent_iter = get_sentences_iter(SIMPLE_TEST_PATH)
    X_test = []
    result = np.array([])
    Y_test = np.array([])
    
    # predict english test samples
    for sentence in en_sent_iter:
        ngrams_vectors = sent_trainer.to_vector(sentence)
        X_test.extend(ngrams_vectors)
        if len(X_test) > MEM_LIMIT:
            Y_test = np.concatenate((Y_test, np.zeros(len(X_test))))
            X_test = np.array(X_test)
            result = np.concatenate((result, sgd.predict(X_test)))
            X_test = []
    
    Y_test = np.concatenate((Y_test, np.zeros(len(X_test))))
    X_test = np.array(X_test)
    result = np.concatenate((result, sgd.predict(X_test)))
    X_test = []
    
    # predict simple test samples
    for sentence in simple_sent_iter:
        ngrams_vectors = sent_trainer.to_vector(sentence)
        X_test.extend(ngrams_vectors)
        if len(X_test) > MEM_LIMIT:
            Y_test = np.concatenate((Y_test, np.ones(len(X_test))))
            X_test = np.array(X_test)
            result = np.concatenate((result, sgd.predict(X_test)))
            X_test = []
    
    Y_test = np.concatenate((Y_test, np.ones(len(X_test))))
    X_test = np.array(X_test)
    result = np.concatenate((result, sgd.predict(X_test)))
    X_test = []

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
    
    
            

    
    