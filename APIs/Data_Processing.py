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
        patternlist1=["(?<=problem description)(.*)(?=(\(2\) site))|(?<=problem description)(.*)"]
        processed = data.copy()
        # print(processed)
        processed['Desc']=processed["Problem Description"]
        processed['Desc'] = processed.Desc.map(lambda x:self.str_lower(str(x)))
        processed['raw'] = processed.Desc.map(lambda x:x.replace("\r",""))
        processed['raw'] = processed.raw.map(lambda x:x.replace("\n"," "))
        processed['raw'] = processed.raw.map(lambda x:x.replace("\t",""))
        processed['raw']=processed.raw.map(lambda x:self.dataextract(x, patternlist1))
        processed['raw'] = processed.raw.map(lambda x:self.remove_blanks(x))
        processed['raw']=processed.raw.map(lambda x:self.dataextract2(x))
        processed['raw']=processed.raw.map(lambda x: re.sub('[,\.!?:()/_-]', ' ', x))
        processed['raw']=processed.raw.map(lambda x:self.dataextract3(x))
        processed.dropna(subset=["raw"], how ='all', inplace = True)
        self.df_reviews = processed.copy()
        self.words = set(nltk.corpus.words.words())
        self.stopwords=stopwords
        lookup = lookup
        self.typo = pd.read_excel(lookup, sheet_name="typo")
        self.typo.words = self.typo.words.map(lambda x: self.removespace_lookup(x))
        self.typo.correction = self.typo.correction.map(lambda x: self.removespace_lookup(x))
        self.worddict = {k:v for k,v in zip(self.typo.words, self.typo.correction)}
        self.df_reviews.raw = self.df_reviews.raw.map(lambda x: self.corr_sent(x))
        self.data_words=self.gen_words(self.df_reviews['raw'])
        self.data_words = self.remove_stopwords(self.data_words, self.stopwords)
        lem = WordNetLemmatizer()
        # print(self.data_words)
        self.df_reviews["desc"]=""
        for i in range(len(self.data_words)):
            text_list = self.data_words[i]
            text_list = [lem.lemmatize(word) for word in text_list]
            text_list = [lem.lemmatize(word, 'v') for word in text_list]
            text_list = [lem.lemmatize(word, 'a') for word in text_list]
            text_list = [lem.lemmatize(word, 'r') for word in text_list]
            text_list = [lem.lemmatize(word, 's') for word in text_list]
            self.df_reviews["desc"].loc[i]= " ".join(text_list)
        self.df_reviews.raw = self.df_reviews.desc.map(lambda x: self.corr_sent(x))
        self.data_words = self.df_reviews.raw.map(lambda x: self.rem_nondict(x))
        self.data_words=self.gen_words(self.data_words)
        self.data_words = self.remove_stopwords(self.data_words, self.stopwords)
        self.df_reviews['common_words'] =  self.data_words
        self.df_reviews['common_words'] = self.df_reviews.common_words.map\
            (lambda x:self.join_words(x))
        self.df_reviews['common_words']=self.df_reviews.common_words.map(lambda x:self.dataextract4(x))
        self.df_reviews['common_words']=self.df_reviews.common_words.map(lambda x:self.dataextract5(x)) 
        self.df_reviews["common_words"].replace(" ",np.nan, inplace=True)
        self.df_reviews.dropna(subset=["common_words"], how ='all', inplace = True)
        
    def getdata(self):
        return [self.df_reviews, self.data_words]
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
        for text in texts:
            new=gensim.utils.simple_preprocess(text,deacc=True)
            final.append(new)
        return (final)
    def remove_stopwords(self,texts, stopwords):
        return [[word for word in simple_preprocess(str(doc)) if word not in \
            stopwords] for doc in texts]
    def rem_nondict(self, sent):
        return  sent #" ".join(w for w in nltk.wordpunct_tokenize(sent) \
                #if w.lower() in self.words or not w.isalpha())
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
        for i in sent:
            for j in i:
        #         print(j)
                if j not in ['error message:', ' ', '*site','','contact',"(2) site"]:
                    s+=" "+"".join(j)
        s=s.strip()
        return s
    def dataextract(self, sent, patternlist):
        return re.findall(patternlist[0], sent)
    def dataextract2(self, sent):
        return re.sub("(?:((on error\*\:*)* (\(including screenshot of error\)( )*\:*)*))"," ", sent).strip()
    def dataextract3(self, sent):
        return re.sub("\b(?![0-9]+\b)(?![a-z]+\b)[0-9a-z]+\b"," ", sent).strip()
    def dataextract4(self, sent):
        return re.sub("(clf)*(wip(.+?)\ )","", sent).strip()
    def dataextract5(self, sent):
        return re.sub("(\s*\w*(infineon)(.+?)\s)|(\s*\w*(ifx)(.+?)\s)", " ", sent).strip() #|(\s*\w*(lot)(.+?)\s)
if __name__ == '__main__':
    print("data processing")
    CONFIG_PATH = "//home//fibebocai//ReactiveProactive//Manna//"
    def load_config(config_name):
        with open(os.path.join(CONFIG_PATH, config_name)) as file:
            config = yaml.safe_load(file)
            return config
    config = load_config("config.yaml")
    config1= config["lda"]
    print("Please provide the excel data name: ")
    data = str(input())
    data = pd.read_excel(os.path.join(config["data_directory"], data), usecols=['Ticketid','Problem Description'])

    text_file = open(os.path.join(config["stopword_directory"], config["stopword_file"]),"r")
    lines = text_file.read().splitlines()
    text_file.close()
    lookup = os.path.join(config["lookup_path"], config["lookup_file"])

    stopwords = stopwords.words("english")
    stopwords.extend(lines)
    a = Data_Processing(data, stopwords, lookup) 
    [df_reviews, data_words] = a.getdata()
    df_reviews.to_excel(os.path.join(config["processedData_directory"], config["processedData_name"]))
        