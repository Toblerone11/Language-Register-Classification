import os
from words_repr import init_word2vec, get_model, get_sentences_iter
from sentences_repr import SentenceTrainer
from sklearn.linear_model import SGDClassifier
import numpy as np
from pprint import pprint
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

BOTH_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\PWKP_108016"
SIMPLE_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\simple_wiki.sentences"
EN_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\en_wiki.sentences"

SIMPLE_TEST_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\code\simple_test.sentences"
EN_TEST_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\en_test.sentences"

WORDS_MODEL_PATH = ".\simple_en_wiki.model"
CLF_PATH = ".\sgdClassifier.pkl"
MEM_LIMIT = 25000

"""
models paths:
    {simple before: ".\simple_en_wiki.model"
     english_before: }
"""


def evaluate_model(y_bar, y):
    TP, TN, FP, FN = 0, 0, 0, 0
    loss = 0
    
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
    print("Loss: ", loss)
    

if __name__ == "__main__":
    init_word2vec(BOTH_PATH, WORDS_MODEL_PATH, forcetrain=False)
    words_model = get_model()
    sent_trainer = SentenceTrainer(words_model)
    forcetrain = True
    # building classifier
    if not os.path.exists(CLF_PATH) or forcetrain:
        sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, fit_intercept=True)
        classes = [0, 1]
        X_en = []
        X_simple = []
        en_sent_iter = get_sentences_iter(EN_PATH)
        simple_sent_iter = get_sentences_iter(SIMPLE_PATH)
        
        # building simple samples
        print("Training Simple corpus")    
        for sentence in simple_sent_iter:
            ngrams_vectors = sent_trainer.to_vector(sentence)
            X_simple.extend(ngrams_vectors)
            if len(X_simple) > MEM_LIMIT:
                X_simple = np.array(X_simple)
                Y_simple = np.ones(len(X_simple))
                sgd.partial_fit(X_simple, Y_simple, classes=classes)
                X_simple = []
        
        X_simple = np.array(X_simple)
        Y_simple = np.ones(len(X_simple))
        sgd.partial_fit(X_simple, Y_simple, classes=classes)
        X_simple = []
        
        # build en samples
        print("Training English corpus")
        for sentence in en_sent_iter:
            ngrams_vectors = sent_trainer.to_vector(sentence)
            X_en.extend(ngrams_vectors)
            if len(X_en) > MEM_LIMIT:
                X_en = np.array(X_en)
                Y_en = np.zeros(len(X_en))
                sgd.partial_fit(X_en, Y_en, classes=classes)
                X_en = []
        
        X_en = np.array(X_en)
        Y_en = np.zeros(len(X_en))
        sgd.partial_fit(X_en, Y_en, classes=classes)
        X_en = []
        
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
    print(score)
    
    
    
    
    
            

    
    