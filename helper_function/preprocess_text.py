import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import pandas as pd
import numpy as nup 
import seaborn as sns
import re
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import WordNetLemmatizer


from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string



def preprocess_text(sentence_list):
  lines_list = []
  stop_words = set(stopwords.words('english'))
  lemmatizer = WordNetLemmatizer()

  for line in sentence_list:
      line = line.lower()
      words=[]
      for word in word_tokenize(line):
        if word.isalpha():
          if word not in stop_words:
            words.append(lemmatizer.lemmatize(word))
      lines_list.append(' '.join(words))
  
  return lines_list

def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize the text into words
    words = word_tokenize(text)
    # Remove stopwords from the text
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Remove unnecessary punctuations
    words = [word for word in words if word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    text = ' '.join(words)
    return text