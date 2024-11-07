import re

from nltk.corpus import stopwords
import jieba
from nltk.tokenize import sent_tokenize, word_tokenize

def preprocess_text(text):
    stop_words = set(stopwords.words('chinese'))
    stop_words = jieba.lcut(text)
    print(stop_words)
    text = re.sub(r'\s+', '', text)
    words = text.split()
    # words = [word for word in words if word not in stop_words]
    return ''.join(words)
