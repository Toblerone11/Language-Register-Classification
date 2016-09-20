import gensim, logging
import sys
import os
import time

# from similarity_space_eval import SimilarityEvaluator

CORPUS_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\simple_wiki.sentences"
MODEL_PATH = 'C:\D\Documents\studies\word2vec\wiki_model'
GOLD_STANDARD_SIMLEX = r"C:\D\Documents\studies\word2vec\gold_simlex.txt"
OUT_PATH = r"C:\D\Documents\studies\word2vec"

# neural network properties
CORPUS_SIZE = 222941
MIN_COUNT = 20
WORKERS = 4
NET_SIZE = 100

# globals
current_model_path = None


def sentence_iterator(path_to_corpus):
    with open(path_to_corpus, 'r', encoding='utf-8', errors='strict') as corpus:
        for sentence in corpus:
            yield sentence.lower().split()


def activate_with_progress(generator, max_size, elements_name):
    """
    this function wraps any other function which needs to be inspected with progress bar and prints progress bar
    to the comand prompt.
    :param generator: a generator type to iterate over its elements,
    :param max_size: the amount of data the generator may yield.
    :param elements_name: str, the name of each element, used for the recording data.
    """
    post_count = 0
    percentage = int(max_size / 100)
    percent_count = 0
    print("num of all %s: %d\npercentage: %d\n" % (elements_name, max_size, percentage))
    sys.stdout.write("\r[%s%s] %d%s" % ('#' * percent_count, ' ' * (100 - percent_count), percent_count, '%'))
    sys.stdout.flush()

    has_next = True
    while has_next:
        try:
            yield generator.__next__()
            post_count += 1
            if post_count == percentage:
                percent_count += 1
                post_count = 0
                sys.stdout.write(
                        "\r[%s%s] %d%s" % ('#' * percent_count, ' ' * (100 - percent_count), percent_count, '%'))
                sys.stdout.flush()
                if percent_count == 62:
                    print('', end='')

                    # if percent_count == 80:
                    #     break
        except StopIteration:
            has_next = False
            pass
    print('\n')


def lines_size(path_to_corpus):
    """
    determines the number of lines in the corpus
    """
    with open(path_to_corpus, 'r', encoding='utf-8', errors='ignore') as c:
        for i, l in enumerate(c):
            pass
        
        return i + 1

def get_sentences_iter(path_to_corpus):
    it = sentence_iterator(path_to_corpus)
    corpus_size = lines_size(path_to_corpus)
    return activate_with_progress(it, corpus_size, "sentences")


def init_word2vec(path_to_corpus, out_model_path=MODEL_PATH, forcetrain=False):
    global current_model_path
    
    if os.path.exists(out_model_path) and not forcetrain:
        current_model_path = out_model_path
        return
        
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("creating an empty model")
    model = gensim.models.Word2Vec(min_count=MIN_COUNT, workers=WORKERS, size=NET_SIZE, hs=1, negative=0)  # an empty model, no training
    print("building the dictionary")
    model.build_vocab(get_sentences_iter(path_to_corpus))  # can be a non-repeatable, 1-pass generator
    print("training the neural net")
    model.train(get_sentences_iter(path_to_corpus))  # can be a non-repeatable, 1-pass generator

    # model = gensim.models.Word2Vec(sentences)
    model.save(out_model_path)
    current_model_path = out_model_path
    

def get_model():
    """
    returns the katest model that was trained using tis library
    """
    return gensim.models.Word2Vec.load(current_model_path)


def init_glove(path_to_corpus):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("creating an empty model")
    model = gensim.models.Word2Vec(min_count=50, workers=4, size=200)  # an empty model, no training
    print("building the dictionary")
    model.build_vocab(get_sentences_iter(path_to_corpus))  # can be a non-repeatable, 1-pass generator
    print("training the neural net")
    model.train(get_sentences_iter(path_to_corpus))  # can be a non-repeatable, 1-pass generator

    # model = gensim.models.Word2Vec(sentences)
    model.save(MODEL_PATH)



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

    

if __name__ == "__main__":
    # init_word2vec(CORPUS_PATH)
    # model = gensim.models.Word2Vec.load(MODEL_PATH)
    # evaluate(model)

    play_with_model()
