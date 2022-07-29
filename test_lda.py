## import general libraries
import pandas as pd
import numpy as np
import nltk
nltk.data.path.append("/home/fibebocai/nltk_data/")
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
import joblib
import os
import yaml
from APIs.Data_Processing_test import Data_Processing
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

"""The oop based class which I created for the purpose of training and testing of the LDA model of sklearn"""
from APIs.sklearnLDA_test import sklearnLDA_test

#####################################################################################
class test_lda:
    def __init__(self, stopwords):
        CONFIG_PATH = "//home//fibebocai//ReactiveProactive//Manna//"
        def load_config(config_name):
            with open(os.path.join(CONFIG_PATH, config_name)) as file:
                config = yaml.safe_load(file)
                return config
        config = load_config("config.yaml")
        config1= config["nmf"]
        print("Please provide your problem statement: ")
        # df_test = pd.read_csv("Data/EAF_data.csv", usecols=['Ticketid','Problem Description']).loc[170,:]
        data = str(input())
        # data= "the fajob : za940359m0xspav0420200908134433329 does not belong to this equipment: sptv01"
        # print(data)
        text_file = open(os.path.join(config["stopword_directory"], config["stopword_file"]),"r")
        lines = text_file.read().splitlines()
        text_file.close()
        lookup = os.path.join(config["lookup_path"], config["lookup_file"])

        stopwords = stopwords.words("english")
        stopwords.extend(lines)
        a = Data_Processing(data, stopwords, lookup) 
        [df_reviews, data_words] = a.getdata()
        # print(df_reviews)
        # df_reviews =  pd.DataFrame([df_reviews])
        # df_reviews = df_reviews[df_reviews['common_words'].apply(lambda x: len(x)>1)]
        # df_reviews.to_excel(os.path.join(config["processedData_directory"], "processedtest_data.xlsx"))
        lda_topics = [str(i) for i in range(config1["num_topics"])]

        # print("model name: ")
        # model = str(input())
        model = "lda"
        ldanmf_modelPath =os.path.join(config["model_directory"], model+".pkl")

        lda_nmf = joblib.load(ldanmf_modelPath)
        # print("vectorizer model name:")
        # vectorizer_model = str(input())
        vectorizer_model = "vectorizer_cvlda"
        vectorizer_cv = joblib.load(os.path.join(config["model_directory"], vectorizer_model+".pkl"))

        # print(df_reviews)
        a = sklearnLDA_test(lda_nmf, vectorizer_cv, df_reviews)
        self.df = a.get_inference(lda_topics, df_reviews,0)
        # print(df)
        # plt.bar(df, lda_topics, color ='maroon')
        # plt.show()
    def result(self):
        return self.df
if __name__ =="__main__":
    CONFIG_PATH = "//home//fibebocai//ReactiveProactive//Manna//"
    def load_config(config_name):
        with open(os.path.join(CONFIG_PATH, config_name)) as file:
            config = yaml.safe_load(file)
            return config
    config = load_config("config.yaml")
    a = test_lda(stopwords)
    df = a.result()
    print(df)
    # df.to_excel(os.path.join(config["testresult_path"], "testresult.xlsx"))