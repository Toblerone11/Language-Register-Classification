import os
import codecs
import random

INPATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\PWKP_108016"
EN_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\en_wiki.sentences"
SIMPLE_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\simple_wiki.sentences"
MIX_PATH = r".\train_set.sentences"
LABEL_PATH = r".\labels.lbl"


def lines_size(path_to_corpus):
    """
    determines the number of lines in the corpus
    """
    with open(path_to_corpus, 'r', encoding='utf-8', errors='ignore') as c:
        for i, l in enumerate(c):
            pass
        
        return i + 1

def randomize_sentences():
    simple_size = lines_size(SIMPLE_PATH)
    en_size = lines_size(EN_PATH)

    simplef = open(SIMPLE_PATH, 'r', encoding='utf-8', errors='ignore')
    enf = open(EN_PATH, 'r', encoding='utf-8', errors='ignore')
    labelf = open(LABEL_PATH, 'w', encoding='utf-8', errors='ignore')
    with open(MIX_PATH, 'w', encoding='utf-8', errors='ignore') as mixf:
        randarr = [1 for _ in range(simple_size)] + [0 for _ in range(en_size)]
        random.shuffle(randarr)
        for i in randarr:
            if i == 0:
                line = enf.readline()
            else:
                line = simplef.readline()

            mixf.write(line)
            labelf.write(str(i))
            labelf.write('\n')


def split_types():
    all_file = codecs.open(INPATH, 'r', encoding='utf-8', errors='ignore')
    en_file = open(EN_PATH, 'w', encoding='utf-8', errors='ignore')
    simple_file = open(SIMPLE_PATH, 'w', encoding='utf-8', errors='ignore')
    
    en_count = 0
    simple_count = 0
    
    line = all_file.readline()
    while (line != ''):
        # get english sentence
        en_sent = line
        en_file.write(en_sent)
        en_count += 1
        # en_file.write(os.linesep)
        
        # get simple sentence
        simple_sent = all_file.readline()
        simple_file.write(simple_sent)
        simple_count += 1
        # simple_file.write(os.linesep)
        
        # check for more simple sentences
        line = all_file.readline()
        while line != '\n':
            simple_file.write(line)
            simple_count += 1
            line = all_file.readline()
            
            # simple_file.write(os.linesep)
        
        try:
            line = all_file.readline()
        except:
            line = unicode(all_file.readline(), 'utf-8')
        
    print("english sentences: ", en_count)
    print("simple sentences: ", simple_count)
    return


if __name__ == "__main__":
    # test()
    # split_types()
    randomize_sentences()
            
