from words_repr import init_word2vec, get_model
import gensim, logging


MIN_COUNT = 20
WORKERS = 4
NET_SIZE = 100



class cbowModel():
    """
    Wraps the scikit-learn SGDClassifier class in order to train with large scale
    dataset, and easily run with different congiurations or datasets.
    """
    def __init__(self, model_path, min_count=MIN_COUNT, workers=WORKERS, size=NET_SIZE, hs=1, negative=0):
        self.model_path = model_path
        self.model1 = None
        self.model2 = None
        self.__sent_tool = sent_tool
        self.classes = None
    
    def fit(self, X):
        pass
        
    def fit_large_scale(self, path_to_X1, path_to_X2, classes):
        """
        fit the model to the given dataset iterator X and the labels iterator y. 
        """
        # creating the first model
        self. model1 = gensim.models.Word2Vec(min_count=MIN_COUNT, workers=WORKERS, size=NET_SIZE, hs=hs, negative=negative) 
        self.model1.build_vocab(get_samples_iterator(path_to_X1))
        self.model1.train(get_samples_iterator(path_to_X1))
        
        # creating the second model
        self. model2 = gensim.models.Word2Vec(min_count=MIN_COUNT, workers=WORKERS, size=NET_SIZE, hs=hs, negative=negative) 
        self.model2.build_vocab(get_samples_iterator(path_to_X2))
        self.model2.train(get_samples_iterator(path_to_X2))
        

    def predict(self, X):
        """
        same as sklearn.SGDClassifier.predict(X)
        """
        pass
        
    def predict_large_scale(self, path_to_class0, path_to_class1, classes):
        """
        use the trained model to predict large dataset X which doesn't fit in single batch.
        X is an iterator of pair tuples where the first index is the sample and the second is the label
        """
        c0_size = lines_size(path_to_class0)
        c1_size = lines_size(path_to_class1)
                
        # check test scores on c0 model space
        c0_iter = get_samples_iterator(path_to_class0)
        c1_iter = get_samples_iterator(path_to_class1)
        c0_scores_on_c0_model = self.model1.score(c0_iter, c0_size)
        c1_scores_on_c0_model = self.model1.score(c1_iter, c1_size)
        
        # check test scores on c1 model space
        c0_iter = get_samples_iterator(path_to_class0)
        c1_iter = get_samples_iterator(path_to_class1)
        c0_scores_on_c1_model = self.model2.score(c0_iter, c0_size)
        c1_scores_on_c1_model = self.model2.score(c1_iter, c1_size)
        
        # evaluate
        c0_results = [classes[c0_scores_on_c0_model[i] < c0_scores_on_c1_model[i]] for i in range(c0_size)]
        c1_results = [classes[c1_scores_on_c0_model[i] >= c1_scores_on_c1_model[i]] for i in range(c1_size)]
        
        labels0 = [classes[0] for _ in range(c0_size)]
        labels1 = [classes[1] for _ in range(c1_size)]
        
        return c0_results + c1_results, labels0 + labels1
        
        