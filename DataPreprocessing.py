from nltk.tokenize import word_tokenize
import io
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import os.path

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

def get_vocab(file_name):
    lexicon = []
    with io.open(file_name, 'r', encoding='cp437') as f:
        contents = f.readlines()
        for l in contents:
            all_words = tokenizer.tokenize(l)
            #all_words = word_tokenize(l)
            lexicon += list(all_words)
    lexicon = [(lemmatizer.lemmatize(i)).encode('utf8') for i in lexicon]
    lexicon = sorted(set(lexicon))
    if not os.path.isfile(file_name+"_vocab"):
        with open(file_name+"_vocab", "w") as f:
            for word in lexicon:
                f.write(word)
                f.write("\n")

    return lexicon

def get_max_senLen(file_name):
    max = 0
    with io.open(file_name, 'r', encoding='cp437') as f:
        contents = f.readlines()
        for l in contents:
            all_words = tokenizer.tokenize(l)
            lexicon = list(all_words)
            if len(lexicon) >= max:
                max = len(lexicon)
    return max