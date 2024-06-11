from unittest import result
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE





def get_data(df):
    
    embeddings = save_embeddings(df['embedding'])
    backdoor_labels = get_backdoor_labels(df['backdoor'])
    labels = df['label']
    return labels, backdoor_labels, embeddings

def save_embeddings(string_array):
    i = 0
    result = []
    for string in string_array:
        list_embedding = convert_to_array(string)
        result.append(list_embedding)
        if i % 500 == 0:
            #print("dimensione np_array: ", len(list_embedding))
            print(i , "iterations")
        i = i+1
    #save embeddings in a file
    return np.array(result)


def convert_to_array(emb_string):
        emb_string = emb_string.replace("tensor", "")
        emb_string = emb_string.replace('[', '')
        emb_string = emb_string.replace(']', '')
        emb_string = emb_string.replace('(', '')
        emb_string = emb_string.replace(')', '')
        return np.fromstring(emb_string, sep=',').tolist()

def get_backdoor_labels(string_array):
    result = []
    i = 0
    j = 0
    for string in string_array:
        if string == 'yes':
            i = i+1
            result.append(1)
        else:
            result.append(0)
            j = j+1
    print("Number of backdoor samples:", i)
    print("Number of cleared sample: ", j)
    result = np.array(result)
    return result
    
def read_dataset():

    df = pd.read_csv("dataset_summary.csv")
    df1 = pd.read_csv("dataset_summary_added_backdoor_data.csv")
    frames = [df, df1]
    df_total = pd.concat(frames)
    
    df_lstm = pd.read_csv("dataset_summary_added_backdoor_data_lstm.csv")

    df_pplm = df_total.iloc[29205:, :]

    df_clear = df_total[df_total['backdoor']=='no']

    frames = [df_clear, df_lstm]
    df_clear_lstm = pd.concat(frames)

    #data = labels, backdoor_labels, embeddings
    data_total_plm = get_data(df_total)
    data_lstm = get_data(df_lstm)
    data_pplm = get_data(df_pplm)
    data_clear = get_data(df_clear)
    data_total_lstm = get_data(df_clear_lstm)

    #visualize
    pca = PCA(n_components=2)
    tsne = TSNE()
    X_embedded_tsne = tsne.fit_transform(data_total_plm[2])
    X_embedded = pca.fit_transform(data_total_plm[2])
    ax = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=data_total_plm[1], legend=["Clear samples", "Backdoor Samples"])
    pl.legend(fontsize=30)
    pl.xlabel("X-axis", fontsize=30)
    pl.ylabel("Y-axis", fontsize=30)
    pl.xticks(fontsize = 25)
    pl.yticks(fontsize = 25)
    pl.show()

    ax = sns.scatterplot(X_embedded_tsne[:,0], X_embedded_tsne[:,1], hue=data_total_plm[1], legend=["Clear samples", "Backdoor Samples"])
    pl.legend(fontsize=30)
    pl.xlabel("X-axis", fontsize=30)
    pl.ylabel("Y-axis", fontsize=30)
    pl.xticks(fontsize = 25)
    pl.yticks(fontsize = 25)
    pl.show()



    X_embedded_tsne = tsne.fit_transform(data_total_lstm[2])
    X_embedded = pca.fit_transform(data_total_lstm[2])

    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=data_total_lstm[1], legend=["Clear samples", "Backdoor Samples"])
    pl.legend(fontsize=30)
    pl.xlabel("X-axis", fontsize=30)
    pl.ylabel("Y-axis", fontsize=30)
    pl.xticks(fontsize = 25)
    pl.yticks(fontsize = 25)
    pl.show()

    sns.scatterplot(X_embedded_tsne[:,0], X_embedded_tsne[:,1], hue=data_total_lstm[1], legend=["Clear samples", "Backdoor Samples"])
    pl.legend(fontsize=30)
    pl.xlabel("X-axis", fontsize=30)
    pl.ylabel("Y-axis", fontsize=30)
    pl.xticks(fontsize = 25)
    pl.yticks(fontsize = 25)
    pl.show()






    return data_total_plm, data_total_lstm, data_clear, data_lstm, data_pplm



