## import general libraries
from lib2to3.pgen2.pgen import DFAState
import numpy as np
import pandas as pd
import re
import nltk
nltk.data.path.append("/home/fibebocai/nltk_data/")
import ray
#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
##spacy
# import spacy
from nltk.corpus import stopwords
import en_core_web_sm
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import joblib
import yaml



class Data_Processing():
    def __init__(self, data, stopwords, lookup):
        # patternlist1=["((problem description)|(issue))(.*)(?=((\(2\) site\W)|(site\W)|(equipment name\W)))"]
        # processed = pd.DataFrame([data], columns=['Desc'])
        # print(processed["Problem Description"])
        # processed['Desc']=processed["Problem Description"]
        processed =str(data).lower()
        # processed['raw'] = processed.raw.replace("\r","")
        # print(processed['raw'])
        # processed['raw'] = processed.raw.replace("\n"," ")
        # processed['raw'] = processed.raw.replace("\t","")
        # print(processed['raw'])
        # processed['raw']=self.dataextract(processed.raw, patternlist1)
        # print(processed['raw'])
        # processed['raw'] = self.remove_blanks(processed.raw)
        # processed['raw']=self.dataextract2(processed.raw)
        processed=re.sub('[,\.!?:()/_-]', ' ', processed)
        processed= self.dataextract3(processed)
        # self.df_reviews = processed.copy()
        self.words = set(nltk.corpus.words.words())
        self.stopwords=stopwords
        lookup = lookup
        self.typo = pd.read_excel(lookup, sheet_name="typo")
        self.typo.words = self.typo.words.map(lambda x: self.removespace_lookup(x))
        self.typo.correction = self.typo.correction.map(lambda x: self.removespace_lookup(x))
        self.worddict = {k:v for k,v in zip(self.typo.words, self.typo.correction)}
        processed = self.corr_sent(processed)
        
        self.data_words=self.gen_words(processed)
        
        # print(pd.DataFrame([self.df_reviews]))
        self.data_words = self.remove_stopwords(self.data_words, self.stopwords)
        # print(self.data_words)
        lem = WordNetLemmatizer()
        # self.df_reviews["desc"]=""
        new =""
        for i in range(len(self.data_words)):
            text_list = self.data_words[i]
            text_list = [lem.lemmatize(word) for word in text_list]
            text_list = [lem.lemmatize(word, 'v') for word in text_list]
            text_list = [lem.lemmatize(word, 'a') for word in text_list]
            text_list = [lem.lemmatize(word, 'r') for word in text_list]
            text_list = [lem.lemmatize(word, 's') for word in text_list]
            new+= " "+" ".join(text_list)
        new=new.strip()
        # print(new)
        # self.df_reviews.raw = self.corr_sent(self.df_reviews.desc)
        self.data_words = self.rem_nondict(new)
        # self.data_words=self.gen_words(self.data_words)
        # self.data_words = self.remove_stopwords(self.data_words, self.stopwords)
        processed =  self.data_words
        # self.df_reviews['common_words'] = self.join_words(self.df_reviews.common_words)
        processed= self.dataextract3(processed)
        processed=self.dataextract4(processed)
        self.processed=self.dataextract5(processed)
        # self.df_reviews['common_words'] = self.remove_blanks(self.df_reviews.common_words)
        # print(self.df_reviews.common_words)
        
    def getdata(self):
        return [self.processed, self.data_words]
    def removespace_lookup(self, word):
        return word.strip()
    def corr_sent(self,sent):
        texts= sent.split(" ")
        for i in texts:
            if i in self.worddict.keys():
                sent = sent.replace(i, self.worddict[i])
        return sent
    def gen_words(self, texts):
        final=[]
        texts = texts.split(" ")
        for text in texts:
            # print(text)
            new=gensim.utils.simple_preprocess(text,deacc=True)
            final.append(new)
        return (final)
    def remove_stopwords(self,texts, stopwords):
        return [[word for word in simple_preprocess(str(doc)) if word not in \
            stopwords] for doc in texts]
    def rem_nondict(self, sent):
        return  " ".join(w for w in nltk.wordpunct_tokenize(sent) \
                if w.lower() in self.words or not w.isalpha())
    def join_words(self,l):
        return " ".join(l)
    def str_lower(self, sent):
        words = sent
        new_word=""
        for word in words:
            new_word+=" ".join(word.lower())
        return new_word
    def remove_blanks(self, sent):
        s=""
        # sent = sent.split(" ")
        for i in sent:
            for j in i:
        #         print(j)
                if j not in ['error message:', ' ', '*site','','contact',"site*"]:
                    s+=" "+"".join(i)   
        s=s.strip()
        return s
    def dataextract(self, sent, patternlist):
        return re.findall(patternlist[0], sent)
    def dataextract2(self, sent):
        return re.sub(r"(?:((on error\*\:*)* (\(including screenshot of error\)( )*\:*)*))"," ", sent).strip()
    def dataextract3(self, sent):
        return re.sub(r"\b(?![0-9]+\b)(?![a-z]+\b)[0-9a-z]+\b"," ", sent).strip()
    def dataextract4(self, sent):
        return re.sub(r"(clf)*(wip(.+?)\ )","", sent).strip()
    def dataextract5(self, sent):
        return re.sub(r"(\s*\w*(infineon)(.+?)\s)|(\s*\w*(ifx)(.+?)\s)|(\s*\w*(lot)(.+?)\s)", " ", sent).strip()
if __name__ == '__main__':
    print("data processing")
