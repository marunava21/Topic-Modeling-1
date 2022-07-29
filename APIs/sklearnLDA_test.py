import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim.corpora as corpora
import pandas as pd
import yaml
import os

class sklearnLDA_test():
    def __init__(self, model, vectorizer_cv, df_reviews):
        # self.num_words = num_words
        self.df_reviews = df_reviews
        self.model = model
        self.CONFIG_PATH = "//home//fibebocai//ReactiveProactive//Manna//"
        config = self.load_config("config.yaml")
        config1= config["nmf"]
        self.vectorizer_cv = vectorizer_cv
    def load_config(self, config_name):
        with open(os.path.join(self.CONFIG_PATH, config_name)) as file:
            config = yaml.safe_load(file)
        return config
    def get_inference(self, lda_topics, df, threshold):
        text = df
        v_text = self.vectorizer_cv.transform([text])
        score = self.model.transform(v_text)
        labels = set()
        for i in range(len(score[0])):
            if score[0][i] > threshold:
                labels.add(lda_topics[i])

        if not labels:
            return 'None', -1, set()
        df_new = pd.DataFrame(np.array(score), columns=[lda_topics])
        Z = [x for _,x in sorted(zip(score[0],lda_topics))][::-1]
        return Z


if __name__ =='__main__':
    print("testing")