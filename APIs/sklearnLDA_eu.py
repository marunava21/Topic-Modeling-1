import numpy as np
from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from fuzzywuzzy import fuzz
import gensim.corpora as corpora
import pandas as pd

class sklearnLDA():
    def __init__(self, df_reviews, ld, alpha, eta,lm, lo, bs, num_topics, num_words):
        self.num_words = num_words
        self.df_reviews = df_reviews
        self.lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10,learning_decay=ld, \
                                        doc_topic_prior=alpha, topic_word_prior=eta, learning_method= lm ,learning_offset=lo,\
                                        n_jobs=-1,\
                                        evaluate_every = -1,batch_size=bs, random_state=100)
        self.vectorizer_cv = TfidfVectorizer(analyzer='word',ngram_range=(1,2))
        self.X_cv = self.vectorizer_cv.fit_transform(self.df_reviews['desc']).toarray()

    def topic_gen(self):
        W1=self.lda.fit_transform(self.X_cv)
        H1=self.lda.components_
        vocab=np.array(self.vectorizer_cv.get_feature_names())
        self.top_words=lambda t: [vocab[i] for i in np.argsort(t)[:-self.num_words-1:-1]]
        self.topic_words=([self.top_words(t) for t in H1])
        self.topics=[' '.join(t) for t in self.topic_words]
        colnames=["Topic"+ str(i) for i in range(self.lda.n_components)]
        df_doc_topic=pd.DataFrame(np.round(W1,2), columns=colnames)
        significant_topic=np.argmax(df_doc_topic.values,axis=1)
        return [self.lda, self.vectorizer_cv, self.X_cv, self.topics, df_doc_topic, significant_topic]
    def table_formatting(self, df_doc_topic):
        df_topic_table=pd.concat([self.df_reviews,df_doc_topic],axis=1)
        df_doc_topic=df_doc_topic.reset_index(drop=True)
        # df_topic_table=df_topic_table.drop('index',axis=1)
        # df_topic_table=df_topic_table.drop('Unnamed: 0',axis=1)
        df_topic_table=df_topic_table.groupby(['Desc']).mean().reset_index()
        return df_topic_table

    def get_inference(self, lda_topics, text, threshold):
        v_text = self.vectorizer_cv.transform([text])
        score = self.lda.transform(v_text)

        labels = set()
        for i in range(len(score[0])):
            if score[0][i] > threshold:
                labels.add(lda_topics[i])

        if not labels:
            return 'None', -1, set()
        return lda_topics[np.argmax(score)]
    def get_model_topics(self, lda_topics, n_top_words):
        word_dict = {}
        feature_names = self.vectorizer_cv.get_feature_names()
        for topic_idx, topic in enumerate(self.lda.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
    #         print(i)
            top_features = [feature_names[i] for i in top_features_ind]
            word_dict[lda_topics[topic_idx]] = top_features

        return pd.DataFrame(word_dict)


if __name__ =='__main__':
    print("LDA model is running----->")