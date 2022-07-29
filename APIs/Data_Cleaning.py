#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from stop_words import get_stop_words
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string 



def datacleaning(rawdata):

    # Import Datasets 🔌
    prob = pd.DataFrame(rawdata, columns = ['Ticketid', 'Problem Description', 'Sub Category', 'Resolution'])
    prob["Problem Desc"] = prob['Problem Description']
    prob.rename(columns = {'Problem Description': 'raw', 'General Category': 'Category'}, inplace = True)

    prob.raw = prob.raw.astype(str).str.lower()
    prob.Resolution = prob.Resolution.astype(str).str.lower()
    prob.raw = prob.raw.astype(str).str.replace('including screenshot of error',' ')
    prob.raw = prob.raw.astype(str).str.replace('(?:[\t ]*(?:\r?\n|\r|\t))+\s*',' ', regex=True)
    
    # Text Extraction ✂️
    patternlist = [["Desc", "lot#", "equipment", "eqp_name","server","desc_temp1","desc_temp2","desc_temp3"],
               ['error(.+?)\(2', 
                '\(3\)\slot\snumber\:(.+?)\(4',
                '\(4\)\sequipments\saffected\:(.+?)\(5',
                '\(5\)\sequipment\/pc\sname\:(.+?)\(6',
                '\(6\)\scamstar\sserver\*\:(.+?)\(7',
                'description: (.+)'
                ,'description(.+?)\.'
                ,'change\ssummary\sto\smost\sfitting\scategory\:(.+?)\sisc'
               ]]
    for i in range(len(patternlist[1])):
        prob[patternlist[0][i]] = prob.raw.astype(str).str.extract(patternlist[1][i])
        
    # Data Cleaning: features impultation & Outliers🔦
    prob.Desc = prob.Desc.fillna(prob.desc_temp1)
    prob.Desc = prob.Desc.fillna(prob.desc_temp2)
    #prob.Desc = prob.Desc.fillna(prob.desc_temp3)
 
    
    prob.Desc = prob.Desc.replace(regex=r'\'', value='') #remove '
    prob.Desc = prob.Desc.replace(regex=[r'\w{14,}', r'\w*\d\w*',r'[^\w\s]',r'\b\d+'], value=' ') #remove punctuation & digits
    prob.Desc = prob.Desc.replace(regex ='\s{2,}', value = " ")
    prob.Resolution = prob.Resolution.replace(regex=[r'\w{14,}', r'\w*\d\w*',r'[^\w\s]',r'\b\d+'], value=' ') #remove punctuation & digits
    prob.Resolution = prob.Resolution.replace(regex ='\s{2,}', value = " ")
    prob = prob.apply(lambda x: x.astype(str).str.strip())
    prob = prob.replace(['','nan'], ' ')
    prob['Sub Category'] = prob['Sub Category'].astype(str).str.capitalize()
#     prob.Category = prob.Category.astype(str).str.capitalize()
#     prob.Site = prob.Site.astype(str).str.capitalize()
    prob = prob.apply(lambda x: x.astype(str).str.strip())
    prob = prob.replace(['','nan'], np.nan)
    #prob['count'] = prob.Desc.str.split().str.len()

    

#     temp = prob.eqp_name.astype(str).str.extract('(isc\w{3,})| (mkz\w{3,})')
#     prob.eqp_name = temp[0].fillna(temp[1])
#     prob.server = prob.server.astype(str).str.extract('(\w+sapa)')
#     prob.equipment = prob.equipment.astype(str).str.extract('(\w.{3,})')
    

    return prob


# In[ ]:


def datamapping(clean_data, fileref):
    lookup = fileref
    prob = clean_data

    #Changing typos and wrong labels 🔦
#     eqpname = pd.read_excel(lookup, sheet_name ='eqplist', usecols=['RESOURCENAME','Facility'])
#     eqpname.RESOURCENAME = eqpname.RESOURCENAME.astype(str).str.lower()
#     prob['Facility'] = prob.equipment.map(eqpname.set_index('RESOURCENAME')['Facility']).astype(str).str.lower()
    
    typo = pd.read_excel(lookup, sheet_name="typo")
    worddict = dict(zip(typo.words, typo.correction))
#     print(typo.words)
    prob.Desc = prob.Desc.replace(worddict, regex=True)
#     prob.Resolution = prob.Resolution.replace(worddict, regex=True)
#     prob["input"] = prob.Desc.fillna('') + " " + prob.Resolution.astype(str).str.lower().fillna('') + " " + prob.Facility.fillna('')

    return prob


def datapreprocessing(prob):

##convert all words to lower ##
#     prob.Desc = prob.Desc.astype(str).str.lower()

# ##remove punctuation ##
#     PUNCT_TO_REMOVE = string.punctuation
#     def remove_punctuation(text):
#         return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
#     prob.Desc = prob.Desc.apply(lambda text: remove_punctuation(text))

## Remove stopwords ##
    STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    # STOPWORDS = set(stopwords.words('english'))
    def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    prob.Desc= prob.Desc.apply(lambda text: remove_stopwords(text))


## Stemming ##
    #stemmer = PorterStemmer()
    #def stem_words(text):
     #   return " ".join([stemmer.stem(word) for word in text.split()])
    #prob.input = prob.input.apply(lambda text: stem_words(text))


    return prob

