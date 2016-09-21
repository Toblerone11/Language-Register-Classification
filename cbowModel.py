from words_repr import init_word2vec, get_model
import gensim, logging
from datatools import get_samples_iterator, lines_size


MIN_COUNT = 20
WORKERS = 4
NET_SIZE = 100

MODEL1_PATH = r".\model1.model"
MODEL2_PATH = r".\model2.model"

class cbowModel():
    """
    Wraps the scikit-learn SGDClassifier class in order to train with large scale
    dataset, and easily run with different congiurations or datasets.
    """
    def __init__(self, model_path, min_count=MIN_COUNT, workers=WORKERS, size=NET_SIZE, hs=1, negative=0):
        self.model_path = model_path
        self.model1 = None
        self.model2 = None
        self.classes = None
        self.hs = hs
        self.negative = negative
    
    def fit(self, X):
        pass
        
    def fit_large_scale(self, cat0, reg0, cat1, reg1, classes):
        """
        fit the model to the given dataset iterator X and the labels iterator y. 
        """
        # self.model1 = gensim.models.Word2Vec.load(MODEL1_PATH)
        # creating the first model
        self. model1 = gensim.models.Word2Vec(min_count=MIN_COUNT, workers=WORKERS, size=NET_SIZE, hs=self.hs, negative=self.negative) 
        self.model1.build_vocab(get_samples_iterator(category=cat0, register=reg0))
        self.model1.train(get_samples_iterator(category=cat0, register=reg0))
        self.model1.save(MODEL1_PATH)
        
        # creating the second model
        # self.model2 = gensim.models.Word2Vec.load(MODEL2_PATH)
        self. model2 = gensim.models.Word2Vec(min_count=MIN_COUNT, workers=WORKERS, size=NET_SIZE, hs=self.hs, negative=self.negative) 
        self.model2.build_vocab(get_samples_iterator(category=cat1, register=reg1))
        self.model2.train(get_samples_iterator(category=cat1, register=reg1))
        self.model2.save(MODEL2_PATH)

    def predict(self, X):
        """
        same as sklearn.SGDClassifier.predict(X)
        """
        pass
        
    def predict_large_scale(self, cat0, reg0, cat1, reg1, classes):
        """
        use the trained model to predict large dataset X which doesn't fit in single batch.
        X is an iterator of pair tuples where the first index is the sample and the second is the label
        """
        c0_size = 21604
        c1_size = 22985
                
        # check test scores on c0 model space
        c0_iter = get_samples_iterator(category=cat0, register=reg0)
        c1_iter = get_samples_iterator(category=cat1, register=reg1)
        c0_scores_on_c0_model = self.model1.score(c0_iter, c0_size)
        c1_scores_on_c0_model = self.model1.score(c1_iter, c1_size)
        
        # check test scores on c1 model space
        c0_iter = get_samples_iterator(category=cat0, register=reg0)
        c1_iter = get_samples_iterator(category=cat1, register=reg1)
        c0_scores_on_c1_model = self.model2.score(c0_iter, c0_size)
        c1_scores_on_c1_model = self.model2.score(c1_iter, c1_size)
        
        # evaluate
        print(len(c0_scores_on_c0_model))
        print(len(c1_scores_on_c0_model))
        c0_results = [classes[c0_scores_on_c0_model[i] >= c0_scores_on_c1_model[i]] for i in range(c0_size)]
        c1_results = [classes[c1_scores_on_c0_model[i] < c1_scores_on_c1_model[i]] for i in range(c1_size)]
        
        labels0 = [classes[0] for _ in range(c0_size)]
        labels1 = [classes[1] for _ in range(c1_size)]
        
        return c0_results + c1_results, labels0 + labels1
        
        