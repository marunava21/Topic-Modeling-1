## import general libraries
import numpy as np
import pandas as pd
import re
# import json
# import glob
import nltk
#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
# from gensim.models import coherencemodel
from Data_Cleaning import datacleaning, datamapping, datapreprocessing
##spacy
# import spacy
from nltk.corpus import stopwords
import en_core_web_sm
from nltk.stem import WordNetLemmatizer
# nltk.download('words')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

##vis
# import pyLDAvis
# import pyLDAvis.gensim_models

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=all)



class Data_Processing():
    def __init__(self, ticketname, stopwords):
        self.df_reviews = pd.read_excel(ticketname)
        self.df_reviews.dropna(subset = ['Desc'], how ='all', inplace = True)
        self.words = set(nltk.corpus.words.words())
        self.df_reviews['Desc'] = self.df_reviews['Desc'].map(lambda x: re.sub('[,\.!?]', '', x))
        self.stopwords=stopwords
        self.fileref = 'lookup.xlsx'
        processed = datamapping(self.df_reviews,self.fileref)
        self.data_words=self.gen_words(processed['Desc'])
        self.data_words = self.remove_stopwords(self.data_words, self.stopwords)
        lem = WordNetLemmatizer()
        self.df_reviews["desc"]=""
        for i in range(len(self.data_words)):
            text_list = self.data_words[i]
            text_list = [lem.lemmatize(word) for word in text_list]
            text_list = [lem.lemmatize(word, 'v') for word in text_list]
            text_list = [lem.lemmatize(word, 'a') for word in text_list]
            text_list = [lem.lemmatize(word, 'r') for word in text_list]
            text_list = [lem.lemmatize(word, 's') for word in text_list]
            self.df_reviews["desc"].loc[i]= " ".join(text_list)
        lookup = self.fileref
        self.typo = pd.read_excel(lookup, sheet_name="typo")
        self.worddict = dict(zip(self.typo.words, self.typo.correction))
        self.df_reviews.desc = self.df_reviews.desc.replace(self.worddict, regex=True)
        self.data_words = self.df_reviews.desc.map(lambda x: self.rem_nondict(x))
        self.data_words=self.gen_words(self.data_words)
        self.data_words = self.remove_stopwords(self.data_words, self.stopwords)
        self.df_reviews['common_words'] =  self.data_words
        self.df_reviews['common_words'] = self.df_reviews.common_words.map\
            (lambda x:self.join_words(x))    
        
    def getdata(self):
        return [self.df_reviews, self.data_words]

    def keep_english(self, x):
        englishwords=" ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in self.words or not w.isalpha())
        return englishwords
    def gen_words(self, texts):
        final=[]
        for text in texts:
            new=gensim.utils.simple_preprocess(text,deacc=True)
            final.append(new)
        return (final)
    def remove_stopwords(self,texts, stopwords):
        return [[word for word in simple_preprocess(str(doc)) if word not in \
            stopwords] for doc in texts]
    def rem_nondict(self, sent):
        return " ".join(w for w in nltk.wordpunct_tokenize(sent) \
                        if w.lower() in self.words or not w.isalpha())
    def join_words(self,l):
        return " ".join(l)
if __name__ == '__main__':
    print("data processing")
    # ticketname = "tickets (6).xlsx"
    # a = Data_Processing(ticketname)
    