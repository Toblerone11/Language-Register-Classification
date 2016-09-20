from itertools import chain
from nltk import ngrams
import numpy as np

class SentenceTrainer():
    def __init__(self, words_model, seq_len=5):
        """
        :param: words_model is a word2vec model for words.
        """
        self.__words_model = words_model
        self.__seq_len=seq_len
        
    def to_ngrams(self, sentence):
        """
        this methods takes the sentence and returne the sentence's ngramss as list of tuples which is 
        """
        return ngrams(sentence, self.__seq_len)
    
    def to_vector(self, sentence):
        """
        takes sentence and returns a list of vectors were each vector represent a sequence of self.__seq_len 
        words in the sentence (default is sequences of 5). the representation of each sequence is concatenation of 
        the word's vectors from the given self.__words_model. order of words is preserved
        """
        words_vecs_seqs = []
        start = 0
        end = 0
        for word in sentence:
            if word in self.__words_model:
                end += 1
                continue
            else:
                if (end - start) >= 5:
                    words_vecs_seqs.append([self.__words_model[token] for token in sentence[start:end]])
                end += 1
                start = end

        words_vecs_seqs.append([self.__words_model[token] for token in sentence[start:end]])
        
            
        result = list(chain(*[list(ngrams(words_vecs, self.__seq_len)) for words_vecs in words_vecs_seqs]))
        result = [np.array(list(chain(*vecs))) for vecs in result]
        # result = [vec for vec in result if len(vec) > 0]
        return result

def test():
    s = "The name April comes from that Latin word aperire which means \"to open\"."
    SentenceTrainer()

if __name__ == "__main__":
    test()
            