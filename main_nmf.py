## import general libraries
import pandas as pd
import nltk
nltk.data.path.append("/home/fibebocai/nltk_data/")
import ray
import joblib
import os
import yaml


"""The oop based class which I created for the purpose of training and testing of the LDA model of sklearn"""
from APIs.sklearnLDA import sklearnLDA_NMF

#####################################################################################

CONFIG_PATH = "//home//fibebocai//ReactiveProactive//Manna//"
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
        return config
config = load_config("config.yaml")
config1= config["nmf"]
print("Please provide the excel processed data name: ")
data = str(input())
df_reviews = pd.read_excel(os.path.join(config["processedData_directory"], data), usecols=['Ticketid','Desc','common_words'])

nmf_topics = [str(i) for i in range(config1["num_topics"])]



##NMF Model
model = "nmf"

b = sklearnLDA_NMF.remote(model,df_reviews, config1["learning_decay"], config1["alpha"], config1["eta"],\
     config1["epochs"], config1["learning_method"], config1["learning_offset"], config1["batch_size"],\
         config1["num_topics"], config1["num_words"], config1["random_state"], (config1["ngram_min"],\
            config1["ngram_max"]),config1["min_df"], config1["max_df"], config1["max_features"])
[nmf, vectorizer_cvNMF, X_cv, topics, df_doc_topic, significant_topic] = ray.get(b.topic_gen.remote())
df_topic_table = ray.get(b.table_formatting.remote(df_doc_topic))
df_topic_table.to_excel("Results/df_topic_table_nmf_%d.xlsx"%config1["title_count"])
ray.get(b.get_model_topics.remote(nmf_topics, config1["n_top_words"])).to_excel("Topics/topics_nmf_%d.xlsx"\
    %config1["title_count"])

##for saving the hyperparameters
h_list = [config1["num_topics"], config1["num_words"], config1["learning_decay"], config1["alpha"],\
     config1["eta"], config1["learning_method"], config1["batch_size"], config1["learning_offset"],\
         config1["epochs"], config1["random_state"], (config1["ngram_min"], config1["ngram_max"]), \
            config1["max_df"],config1["min_df"], config1["max_features"]]
columns = ["num_topics", "num_words","learning_decay", "alpha", "eta", "learning_method",  "batch_size","learning_offset", \
           "epochs",  "random_state", "ngram_range","max_df", "min_df", "max_features","lda","nmf"]


##Hyperparameter Saving
import csv
h_list.append("topic_lda_%d"%config1["title_count"])
h_list.append("topic_nmf_%d"%config1["title_count"])
f = open('hyperparameters.csv', 'a', newline='')
writer= csv.writer(f)
writer.writerow(h_list)
f.close()

##Saving the model
joblib.dump(vectorizer_cvNMF, os.path.join(config["model_directory"], "vectorizer_cvnmf.pkl"))
joblib.dump(nmf, os.path.join(config["model_directory"], config["model_name_nmf"]))