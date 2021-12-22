import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
import spacy

# Initialize NLP parameters
#nltk.download('popular')

# Cleaning function for new question
def remove_pos(nlp, x, pos_list):
    """NLP cleaning function based on the POS-Tagging of the Spacy library. 
    It allows you to keep only the Parts Of Speech listed as a parameter. 
    Parameters
    ----------------------------------------
    nlp : spacy pipeline
        Load pipeline with options.
        ex : spacy.load('en', exclude=['tok2vec', 'ner', 'parser', 
                                'attribute_ruler', 'lemmatizer'])
    x : string
        Sequence of characters to modify.
    pos_list : list
        List of POS to conserve.
    ----------------------------------------
    """
    # Test of language detection
    lang = detect(x)
    if(lang != "en"):
        # Deep translate if not in English
        x = GoogleTranslator(source='auto', target='en').translate(x)
    
    doc = nlp(x)
    list_text_row = []
    for token in doc:
        if(token.pos_ in pos_list):
            list_text_row.append(token.text)
    join_text_row = " ".join(list_text_row)
    join_text_row = join_text_row.lower().replace("c #", "c#")
    return join_text_row

def text_cleaner(x, nlp, pos_list, lang="english"):
    """Function allowing to carry out the preprossessing on the textual data. 
        It allows you to remove extra spaces, unicode characters, 
        English contractions, links, punctuation and numbers.
        
        The re library for using regular expressions must be loaded beforehand.
        The SpaCy and NLTK librairies must be loaded too. 
    Parameters
    ----------------------------------------
    nlp : spacy pipeline
        Load pipeline with options.
        ex : spacy.load('en', exclude=['tok2vec', 'ner', 'parser', 
                                'attribute_ruler', 'lemmatizer'])
    x : string
        Sequence of characters to modify.
    pos_list : list
        List of POS to conserve.
    ----------------------------------------
    """
    # Remove POS not in "NOUN", "PROPN"
    x = remove_pos(nlp, x, pos_list)
    # Case normalization
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub('\s+', ' ', x)
        
    # Tokenization
    x = nltk.tokenize.word_tokenize(x)
    # List of stop words in select language from NLTK
    stop_words = stopwords.words(lang)
    # Remove stop words
    x = [word for word in x if word not in stop_words 
         and len(word)>2]
    # Lemmatizer
    wn = nltk.WordNetLemmatizer()
    x = [wn.lemmatize(word) for word in x]
    
    # Return cleaned text
    return x
