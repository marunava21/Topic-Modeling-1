import ray
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim.corpora as corpora
import pandas as pd


@ray.remote
class sklearnLDA_NMF():
    def __init__(self, model, df_reviews, ld, alpha, eta,epochs, lm, lo, bs, num_topics, num_words, random_state, ngram_range,\
         min_df, max_df, max_features):
        self.num_words = num_words
        df_reviews['common_words'] = df_reviews.common_words.map(lambda x:str(x))
        self.df_reviews = df_reviews
        if model == "lda":
            self.model = LatentDirichletAllocation(n_components=num_topics, max_iter=epochs,doc_topic_prior=alpha,\
                                            learning_method= lm ,batch_size=bs, random_state=random_state) 
                                            # learning_offset=lo, evaluate_every = -1,doc_topic_prior=alpha, learning_decay=ld, 
        elif model == "nmf":
            self.model = NMF(n_components=num_topics, random_state = random_state,\
                alpha = 0.01, l1_ratio = 1, init = 'random', max_iter = epochs, solver="cd", tol = 0.0001,\
                    verbose=0)
        self.vectorizer_cv = TfidfVectorizer(analyzer='word',ngram_range=ngram_range, \
            max_df=max_df, max_features = max_features)
        
        self.X_cv = self.vectorizer_cv.fit_transform(self.df_reviews['common_words']).toarray()
    
    
    def topic_gen(self):
        W1=self.model.fit_transform(self.X_cv)
        H1=self.model.components_
        vocab=np.array(self.vectorizer_cv.get_feature_names())
        self.top_words=lambda t: [vocab[i] for i in np.argsort(t)[:-self.num_words-1:-1]]
        self.topic_words=([self.top_words(t) for t in H1])
        self.topics=[' '.join(t) for t in self.topic_words]
        colnames=["Topic"+ str(i) for i in range(self.model.n_components)]
        df_doc_topic=pd.DataFrame(np.round(W1,2), columns=colnames)
        significant_topic=np.argmax(df_doc_topic.values,axis=1)
        return [self.model, self.vectorizer_cv, self.X_cv, self.topics, df_doc_topic, significant_topic]
    
    def table_formatting(self, df_doc_topic):
        self.df_reviews1 = self.df_reviews.reset_index(drop=True)
        df_topic_table=pd.concat([self.df_reviews1,df_doc_topic],axis=1)
        df_topic_table=df_topic_table.groupby(['Ticketid', 'Desc']).mean().reset_index()
        return df_topic_table

    def get_inference(self, lda_topics, text, threshold):
        v_text = self.vectorizer_cv.transform([text])
        score = model.transform(v_text)
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
        for topic_idx, topic in enumerate(self.model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            word_dict[lda_topics[topic_idx]] = top_features

        return pd.DataFrame(word_dict)


if __name__ =='__main__':
    print("LDA model is running----->")
    ray.shutdown()
    ray.init(num_cpus = 16)