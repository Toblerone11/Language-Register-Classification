import os
from words_repr import init_word2vec, get_model, get_sentences_iter
from sentences_repr import SentenceTrainer
from sklearn.linear_model import SGDClassifier
import numpy as np
from pprint import pprint
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

BOTH_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\PWKP_108016"
TRAIN_PATH = r"./PWKP_108016/train_set.sentences"
LABEL_PATH = r"./PWKP_108016/labels.lbl"
SIMPLE_TEST_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\code\simple_test.sentences"
EN_TEST_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\en_test.sentences"

WORDS_MODEL_PATH = ".\simple_en_wiki.model"
CLF_PATH = ".\sgdClassifier.pkl"
MEM_LIMIT = 25000

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
    
    print("TP: {TP}\tFP: {FP}\tFN: {FN}\t".format(TP=TP, FP=FP, FN=FN))
    
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
    init_word2vec(BOTH_PATH, WORDS_MODEL_PATH, forcetrain=False)
    words_model = get_model()
    sent_trainer = SentenceTrainer(words_model)
    forcetrain = True
    # building classifier
    if not os.path.exists(CLF_PATH) or forcetrain:
        sgd = SGDClassifier(loss='log', penalty='l1', alpha=0.01, fit_intercept=True)
        classes = [0, 1]
        X_train = []
        y_train = []
        train_iter = get_sentences_iter(TRAIN_PATH)
        labels_iter = get_next_label(LABEL_PATH)

        # building train samples
        print("Training model")    
        for sentence in train_iter:
            ngrams_vectors = sent_trainer.to_vector(sentence)
            X_train.extend(ngrams_vectors)
            label = next(labels_iter)
            y_train.extend(np.array([label for _ in range(len(ngrams_vectors))]))
            if len(X_train) > MEM_LIMIT:
                X_train = np.array(X_train)
                sgd.partial_fit(X_train, y_train, classes=classes)
                X_train = []
                y_train = []
        
        # X_simple = np.array(X_simple)
        # Y_simple = np.ones(len(X_simple))
        # sgd.partial_fit(X_simple, Y_simple, classes=classes)
        # X_simple = []
        
        # build en samples
        # print("Training English corpus")
        # for sentence in en_sent_iter:
        #     ngrams_vectors = sent_trainer.to_vector(sentence)
        #     X_train.extend(ngrams_vectors)
        #     y_train.extend(np.zeros(len(ngrams_vectors)))
            # if len(X_en) > MEM_LIMIT:
            #     X_en = np.array(X_en)
            #     Y_en = np.zeros(len(X_en))
            #     sgd.partial_fit(X_en, Y_en, classes=classes)
            #     X_en = []
        
        # X_en = np.array(X_en)
        # Y_en = np.zeros(len(X_en))
        # sgd.partial_fit(X_en, Y_en, classes=classes)
        # X_en = []
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