import os
import codecs

INPATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\PWKP_108016"
EN_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\en_wiki.sentences"
SIMPLE_PATH = r"C:\D\Documents\studies\cs\mean_comp\final project\corpora\wiki\simple_wiki.sentences"

def restructure_file

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

def test():
    all_file = open(INPATH, 'r')
    print(all_file.readline())
    print(all_file.readline())
    print(all_file.readline())
    line = all_file.readline()
    if line == '\n':
        print(True)
    
    return

if __name__ == "__main__":
    # test()
    split_types()
            