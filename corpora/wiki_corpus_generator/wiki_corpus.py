"""
This module creates corpus from wikipedia documents for register classification.
"""
import sys
import time
from queue import Queue
import re

import wikipedia as wiki
import requests

# wikipedia languages
SIMPLE = "simple"
ENGLISH = "en"

# request pattern
WIKI_REQ_PATT = "https://{lang}.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:{title}&format=json&cmlimit={limit}&cmtype=page|subcat"
WIKI_SWITCH_LANG_REQ = "https://simple.wikipedia.org/w/api.php?action=query&titles={title}&prop=langlinks&lllimit=100&llprop=url&lllang={lang}&format=json"

# file defaults
FILE_NAME_PATT = "{category}_{lang}.sentences"
LIMIT = 1000000  # limit the number of sentences per file

# response variables
QUERY = "query"
CAT_MEM = "categorymembers"
TITLE = "title"

# wiki types
CAT_PAGE = "Category"
TYPE_SEP = ':'

# defaults
DEFAULT_CATEGORIES = ["Politics", "Money", "Sports", "Economics", "Arts", "Technology", "Fashion", "Crime"]
DEFAULT_LANG = ['en']

# content patterns
TITLE_PATT = re.compile("[=]+[^=]+[=]+")
LINESEP_PATT = re.compile("(?:\.|\?|\!)[\W]")
PAR_PATT = re.compile("\([^\)]+\)")
DOTS_PATT = re.compile("[\.]{2,}")
QUOTE = "<!Q>"
NAME = "<!N>"
SEE_ALSO = "== See also =="
NAME_PATT = re.compile("(?:(?:[A-Z]\.[ ]?)*[A-Z][\w]+ (of )?)*(?:[A-Z]\.[ ]?)*(?!^)[A-Z][\w]+")

LIMIT_PASSED = 1
KEEP_COLLECT = 0
RECOVER_TIME = 10


def make_line_replacements(line):
    line = TITLE_PATT.sub('', line)
    line = DOTS_PATT.sub('', line)
    line = PAR_PATT.sub('', line)
    return line

def make_sentence_replacements(sentence):
    ##  need to replace quotes with <!Q>
    line = NAME_PATT.sub(NAME, sentence)
    return line
    
def take_sentences_from_wikidoc(wiki_page):
    content_lines = None
    try:
        content_lines = wiki_page.content.split('\n')
    except:
        return
        
    for line in content_lines:
        if line == SEE_ALSO:
            return
        line = make_line_replacements(line)
        if line == '':
            continue
            
        sentences = LINESEP_PATT.split(line)
        for sentence in sentences:
            sentence = make_sentence_replacements(sentence)
            yield sentence
        

def dump_content(fd, wiki_page):
    sent_num = 0
    for sentence in take_sentences_from_wikidoc(wiki_page):
        if len(sentence.split(" ")) < 5:
            continue
        fd.write(sentence)
        fd.write('\n')
        sent_num += 1
    
    return sent_num

def printTitle(title, prefix=''):
    try:
        print(prefix, title)
    except:
        for ch in title:
            try:
                print(ch, end='')
            except:
                print("!_!", end='')
        print()


def handleDisambiguation(title, lang):
    """
    activated in cases where disambiguation occured.
    detect the most relevant page (checks in the simple wikipedia).
    :return: WikipediaPage object of the page that was chosen
    """
    langlink_pages = requests.get(WIKI_SWITCH_LANG_REQ.format(title=title, lang=lang)).json()[QUERY]["pages"]
    pid = list(langlink_pages.keys())[0]
    try:
        fixed_title = langlink_pages[pid]["langlinks"][0]['*']
        try:
            wpage = wiki.page(fixed_title)
        except:
            return None
    except KeyError:
        return None
    except wiki.exceptions.DisambiguationError:
        return None
    
    return wpage
    

def get_wiki_response(wiki_req):
    wiki_resp = None
    try:
        wiki_resp = requests.get(wiki_req).json()
    except:
        print("########################### ERRoR ###################################")
        failed = True
        while failed:
            time.sleep(RECOVER_TIME)
            try:
                wiki_resp = requests.get(wiki_req).json()
                failed= False
            except:
                continue
    
    return wiki_resp
                
                
def retrieve_contents(category, limit=1000):
    """
    sent request with the given category and returns two lists, one of pages and second of sub-categories
    """
    # filter the pages by those which only appears in the simple english
    wiki_req = WIKI_REQ_PATT.format(lang=SIMPLE, title=category, limit=limit)
    wiki_resp = get_wiki_response(wiki_req)
        
    pages = []
    sub_categories = []
    for page in wiki_resp[QUERY][CAT_MEM]:
        page_titles = page[TITLE].split(TYPE_SEP)
        if len(page_titles) > 1:
            if page_titles[0] == CAT_PAGE: # the page is of some type not regular.
                sub_categories.append(page_titles[1])
        else:
            pages.append(page_titles[0])
    
    return pages, sub_categories

def write_contents(lang_fd_map, lang_limit_map, category):
    """
    Starts from the given category and in top-down way, using BFS,
    parse the content of all pages under the given category and its sub-categories at all levels.
    Then writing the sentences in the related files up to the given limit.
    :param: lang_fd_map - dictionary between language prefix and the fd which stores the sentences of the given top category in that languages
    :param: lang_limit_map - dictionary between language prefix and the limit to the file which relates to that languages.
    :param: category - the top category to start digging pages and more sub-categories
    """
    # initial queue of categories with the given category
    catQ = Queue()
    catQ.put(category)
    while not catQ.empty():
        cat = catQ.get()
        printTitle(cat, prefix="category: ")
        pages, sub_cats = retrieve_contents(cat)
        [catQ.put(cat) for cat in sub_cats]
        for lang in lang_fd_map:
            if lang not in lang_limit_map:
                continue
            wiki.set_lang(lang)
            print("\tlanguage: ", lang)
            for wiki_page in pages:
                printTitle(wiki_page, prefix="\t\tpage: ")
                try:
                    wpage = wiki.page(wiki_page)
                except wiki.exceptions.DisambiguationError:
                    wpage = handleDisambiguation(wiki_page, lang)
                    if wpage is None:
                        continue
                except:
                    continue
                
                n_sentences = dump_content(lang_fd_map[lang], wpage)
                lang_limit_map[lang] -= n_sentences
                if lang_limit_map[lang] <= 0:
                    lang_fd_map[lang].close()
                    del lang_limit_map[lang]
                    break
        
        if len(lang_limit_map) == 0:
            break
    
        
def write_contents_dfs(lang_fd_map, lang_limit_map, pages, categories):
    for lang in lang_fd_map:
        if lang not in lang_limit_map:
            continue    
            
        print("\tlanguage: ", lang)
        sent_num = 0
        wiki.set_lang(lang)
        for title in pages:
            sent_num += dump_content(lang_fd_map[lang], wpage)
            if sent_num >= lang_limit_map[lang]:
                lang_fd_map[lang].close()
                del lang_limit_map[lang]
                break
        
            lang_limit_map[lang] -= sent_num

    if len(lang_limit_map) == 0:
        return LIMIT_PASSED
    
    for cat in categories:
        printTitle(cat, prefix="category: ")
        pages, sub_cats = retrieve_contents(cat)
        status = write_contents(lang_fd_map, lang_limit_map, pages, sub_cats)
        if status == LIMIT_PASSED:
            return LIMIT_PASSED
    
    return KEEP_COLLECT
    
def create_corpus(out_dir, categories=DEFAULT_CATEGORIES, limit=LIMIT, languages=DEFAULT_LANG):
    """
    Generate files containing sentences from wikipedia documents under the given categories.
    For each category, there is a seperated file being created.
    """
    for category in categories:
        print("start: ", category)
        lang_fd_map = {}
        lang_limit_map = {}
        for lang in languages:
            cat_file = FILE_NAME_PATT.format(category=category, lang=lang)
            lang_fd_map[lang] = open(cat_file, 'w', encoding='utf-8', errors='ignore')
            lang_limit_map[lang] = LIMIT
        
        write_contents(lang_fd_map, lang_limit_map, category)
        print("returned")
        print()


if __name__ == "__main__":
    dir_path = r".\\"
    categories = sys.argv[1:]
    if len(categories) == 0:
        categories = DEFAULT_CATEGORIES
    create_corpus(dir_path, categories=categories, languages=[SIMPLE, ENGLISH])