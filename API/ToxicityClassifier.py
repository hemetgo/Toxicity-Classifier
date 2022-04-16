
# import packages
import pickle

import string
import re
import nltk
#from nltk.corpus import stopwords
#from nltk.stem.snowball import SnowballStemmer
#snow_stemmer = SnowballStemmer(language='english')
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
#import en_core_web_sm
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#from sklearn.tree import DecisionTreeClassifier



# CREATE THE CLEAN STRING FUNCTION
nlp = spacy.load('en_core_web_sm')
def clean_string(text, stem="None"):

    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', '', text)

    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im', 'rt', 'u', 'user', 'n', 'youre', 'ð', '']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove mentions
    text_filtered = [re.sub("@[A-Za-z0-9_]+","", w) for w in text_filtered]
    
    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]
    text_filtered = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer() 
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string

def get_encoded(x):
    x = clean_string(x)
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    x = tokenizer.texts_to_sequences([x])
    x = pad_sequences(x, maxlen=120, padding = 'post')
    return x


def predict(sentence):
    sentence = clean_string(sentence)
    sentence = get_encoded(sentence)
    tree = pickle.load(open('tree.pkl', 'rb'))
    return tree.predict(sentence)

print(predict("banana fish"))