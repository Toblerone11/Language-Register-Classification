from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import numpy as np
from sentences_repr import SentenceTool
from datatools import get_samples_iterator

class sgdModel():
    """
    Wraps the scikit-learn SGDClassifier class in order to train with large scale
    dataset, and easily run with different congiurations or datasets.
    """
    def __init__(self, sent_tool, loss='hinge', learnrate=0.01, penalty='l2', reg_factor=0.0001):
        self.sgd = SGDClassifier(loss=loss, alpha=reg_factor, penalty=penalty, learning_rate =learnrate)
        self.__sent_tool = sent_tool
        self.classes = None
    
    def fit(self, X):
        """
        same as sklearn.SGDClassifier.fit(X, y)
        """
        self.sgd.fit(X, y)
        
    def fit_large_scale(self, X, y, classes):
        """
        fit the model to the given dataset iterator X and the labels iterator y. 
        """
        self.classes = classes
        X_train = np.array([])
        y_train = np.array([])
        for sentence in X:
            ngrams_vectors = self.__sent_tool.to_vector(sentence)
            np.append(X_train, ngrams_vectors)
            label = next(y)
            np.append(y_train, np.array([label for _ in range(len(ngrams_vectors))]))
            if len(X_train) > MEM_LIMIT:
                X_train = np.array(X_train)
                sgd.partial_fit(X_train, y_train, classes=classes)
                X_train = np.array([])
                y_train = np.array([])
    
    def predict(self, X):
        """
        same as sklearn.SGDClassifier.predict(X)
        """
        return self.sgd.predict(X)
        
    def predict_large_scale(self, X):
        """
        use the trained model to predict large dataset X which doesn't fit in single batch.
        X is an iterator of pair tuples where the first index is the sample and the second is the label
        """
        results = []
        labels = []
        for sentence, label in X:
            ngrams_vectors = sent_trainer.to_vector(sentence)
            decisions = self.sgd.decision_function(ngrams_vectors)
            if sum(decisions) > 0:
                result = self.classes[1]
            else:
                result = self.classes[0]
            
            results.append(result)
            labels.append(label)
        
        return results, labels
    
    
    
    
    
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