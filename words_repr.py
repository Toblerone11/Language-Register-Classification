
import sys
import os
import time
from datatools import get_samples_iterator

# from similarity_space_eval import SimilarityEvaluator

# neural network properties
CORPUS_SIZE = 222941
MIN_COUNT = 20
WORKERS = 4
NET_SIZE = 100

# globals
current_model_path = None


def init_word2vec(path_to_corpus, out_model_path=MODEL_PATH, forcetrain=False):
    global current_model_path
    
    if os.path.exists(out_model_path) and not forcetrain:
        current_model_path = out_model_path
        return
        
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("creating an empty model")
    model = gensim.models.Word2Vec(min_count=MIN_COUNT, workers=WORKERS, size=NET_SIZE, hs=1, negative=0)  # an empty model, no training
    print("building the dictionary")
    model.build_vocab(get_samples_iterator(path_to_corpus))  # can be a non-repeatable, 1-pass generator
    print("training the neural net")
    model.train(get_samples_iterator(path_to_corpus))  # can be a non-repeatable, 1-pass generator

    # model = gensim.models.Word2Vec(sentences)
    model.save(out_model_path)
    current_model_path = out_model_path
    

def get_model():
    """
    returns the katest model that was trained using tis library
    """
    return gensim.models.Word2Vec.load(current_model_path)


def play_with_model():
    model = gensim.models.Word2Vec.load(MODEL_PATH)
    print("similarity \t plant - tree:\t" + str(model.similarity('plant', 'tree')))
    print("similarity \t he - she:\t\t" + str(model.similarity('he', 'she')))
    print("similarity \t tree - flower:\t" + str(model.similarity('tree', 'flower')))
    print("most similar \t woman + he - man:\t" + str(model.most_similar(positive=['woman', 'he'], negative=['man'])))
    print("most similar \t it: \t\t" + str(model.most_similar('it')))
    print("most similar \t she: \t\t" + str(model.most_similar('she')))
    print("most similar \t he: \t\t" + str(model.most_similar('he')))
    print("most similar \t like: \t\t" + str(model.most_similar('like')))
    print("most similar \t woman + mother - man:\t" + str(model.most_similar(positive=['woman', 'mother'], negative=['man'])))
    print("most similar \t woman + king - man:\t" + str(model.most_similar(positive=['woman', 'king'], negative=['man'])))
    print("most similar \t king: \t\t" + str(model.most_similar('king')))
    print("most similar \t prince: \t\t" + str(model.most_similar('prince')))
    print("most similar \t woman + prince - man:\t" + str(model.most_similar(positive=['woman', 'prince'], negative=['man'])))
    print("most similar \t king + woman - queen:\t" + str(model.most_similar(negative=['queen'], positive=['woman', 'king'])))



def get_gold_similarity(path_to_gold):
    sim_list = []
    file = open(path_to_gold, "r")
    for line in file:
        sim_line = line.split()
        w1 = sim_line[0]
        w2 = sim_line[1]
        pos = sim_line[2]
        score = float(sim_line[3])
        sim_list.append((w1, w2, pos, score))
    file.close()
    return sim_list


def evaluate(model):
    gold_similarity = get_gold_similarity(GOLD_STANDARD_SIMLEX)
    evaluator = SimilarityEvaluator()
    evaluator.set_gold_std(gold_similarity)
    evaluator.set_model_func(model.similarity)
    evaluator.evaluate()

    
# main part only for testing purposes
if __name__ == "__main__":
    # init_word2vec(CORPUS_PATH)
    # model = gensim.models.Word2Vec.load(MODEL_PATH)
    # evaluate(model)

    play_with_model()
