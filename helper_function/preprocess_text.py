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