"""
corpus tools.
usful to iterate over the data.
"""
import random

SIMPLE_TOKENIZE = 'simple'


parallel_en_path = None
parallel_simple_path = None
crime_en_path = None
crime_simple_path = None
sports_en_path = None
sports_simple_path = None
tech_en_path = None
tech_simple_path = None
arts_en_path = None
arts_simple_path = None

EN_PATHS = [parallel_en_path, crime_en_path, sports_en_path, tech_en_path, arts_en_path]
SIMPLE_PATHS = [parallel_simple_path, crime_simple_path, sports_simple_path, tech_simple_path, arts_simple_path]

paths_map = {
                    'all' : {
                            'en' : EN_PATHS,
                            'simple' : SIMPLE_PATHS,
                            'both' : EN_PATHS + SIMPLE_PATHS
                             },
                    'crime' : {
                                'en'  : crime_en_path,
                                'simple': crime_simple_path
                                'both' : [crime_en_path, crime_simple_path]
                                }

                     }

EN_LABEL = 0
SIMPLE_LABEL = 1
EN_STR = 'en'
SIMPLE_STR = 'simple'

def configure(path_to_conf):
    """
    parse configure file which contains paths to all of the corpora files, or 
    part of them.
    contains more details such as indices for labels (classes)
    """
    pass
    
#repeatable generator example 
"""
def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen

@multigen
def myxrange(n):
   i = 0
   while i < n:
     yield i
     i += 1
m = myxrange(5)
print list(m)
print list(m)
'"""


def lines_size(path_to_corpus):
    """
    determines the number of lines in the corpus
    """
    with open(path_to_corpus, 'r', encoding='utf-8', errors='ignore') as c:
        for i, l in enumerate(c):
            pass
        
        return i + 1


def sentence_iterator(path_to_corpus, choose_lines=None, tokenizer=simple_tokenizer, include_tag=False):
    """
    generate sentences from corpus.
    """
    label = None
    if SIMPLE_STR in path_to_corpus:
        label = SIMPLE_LABEL
    else:
        label = EN_LABEL
        
    if choose_lines is None:
        choose_lines = [1 for _ in range(lines_size(path_to_corpus))]
        
    with open(path_to_corpus, 'r', encoding='utf-8', errors='strict') as corpus:
        line = 0
        for sentence in corpus:
            if choose_lines[line]:
                yield tokenizer(sentence.lower()) if not include_tag else (tokenizer(sentence.lower()), label)
            
            line += 1


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


def sentence_multi_iterator(iters):
    """
    generate sentences from the corpus files in 'paths' in randim way.
    """
    first_iter = iters[0]
    rest_iters = iters[1:]
    for sentence in first_iter:
        yield sentence
        for iter in rest_iters::
            yield next(iter)
    

def get_samples_iterator(category='all', register='both',  include_tag=False, equal=True, randomize=True, tokenize='simple', show_progress=True)
    """
    return iterator of samples by the given properties.
    if register is 'both' or category is 'all' then the order will be randomized anyway.
    """
    result_iter = None
    tokenizer = get_tokenizer(name=tokenize)
    paths = paths_map[category][register]
    if not isinstance(paths, list):
        size = lines_size(paths)
        result_iter = sentence_iterator(paths, include_tag=include_tag, tokenizer=tokenize)
    else:
        sizes = [lines_size(path) for path in paths]
        min_size = min(sizes)
        iters = []
        for i in range(len(paths)):
            lines_choose_array = [1 for _ in range(min_size)] + [0 for _ in range(sizes[i] - min_size)]
            random.shuffle(lines_choose_array)
            iters.append(sentence_iterator(paths[i], lines_choose_array, include_tag=include_tag, tokenizer=tokenize)
        
        size = min_size
        result_iter = sentence_multi_iterator(iters)
    
    if show_progress:
        result_iter = activate_with_progress(result_iter, size, "sentences")
    
    return result_iter


def get_sentences_iter(path_to_corpus):
    it = sentence_iterator(path_to_corpus)
    corpus_size = lines_size(path_to_corpus)
    return activate_with_progress(it, corpus_size, "sentences")

        
def simple_tokenize(sentence):
    """
    tokenizer which just splitting the sentence by spaces
    """
    return sentence.split(" ")
    

def custom_tokenize(sentence, key=lambda x: simple_tokenize(x)):
    """
    tokenizer which works by the given key function
    """
    return key(sentence)


def get_tokenizer(name='simple'):
    if name is  SIMPLE_TOKENIZE:
        return simple_tokenize

    


    