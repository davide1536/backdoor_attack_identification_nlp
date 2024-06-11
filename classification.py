from tabnanny import verbose
from read_dataset import read_dataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report, recall_score, precision_score
from sklearn.svm import SVC, OneClassSVM
from sklearn.utils import class_weight
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import backend as K
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier


'''
Perform classification on dataset's embedding using svc and mlp classifier.
'''


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def conf_matrix(y_test, pred_test):    
    
    # Creating a confusion matrix
    con_mat = confusion_matrix(y_test, pred_test)
    con_mat = pd.DataFrame(con_mat, range(2), range(2))
   
    #Ploting the confusion matrix
    plt.figure(figsize=(6,6))
    sns.set(font_scale=3.0) 
    sns.heatmap(con_mat, annot=True, annot_kws={"size": 37}, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel("X-axis", fontsize = 30)
    plt.ylabel("y-axis", fontsize = 30)

    plt.show()


def percentage_plot(backdoor_lables):
    g = sns.barplot(backdoor_labels, backdoor_labels, palette='Set1', estimator=lambda x: len(x) / len(backdoor_labels) )
    


    #Anotating the graph
    for p in g.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            g.text(x+width/2, 
                y+height, 
                '{:.0%}'.format(height), 
                horizontalalignment='center',fontsize=30)

    #Setting the labels
    plt.legend(fontsize = 30)
    plt.xlabel('Class', fontsize=30)
    plt.ylabel('Precentage', fontsize=30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.title('Percentage of backdoored data', fontsize=35)

    plt.show()



def svc(X_train, y_train):
    #SVC balanced (automatic balancing). The balancing was necessary because of the unbalanced dataset
    clf = SVC(class_weight='balanced')
    clf.fit(X_train, y_train)
    pred_test = clf.predict(X_test)
    f1_test = f1_score(y_test, pred_test)
    print('The f1 score for the testing data (balanced):', f1_test)
    conf_matrix(y_test, pred_test)
    print(classification_report(y_test, pred_test))

    #test svc on lstm data
    pred_test = clf.predict(embeddings_lstm)
    f1_test = f1_score(backdoor_labels_lstm, pred_test)
    print('The f1 score (svc) for the testing lstm data (balanced):', f1_test)
    conf_matrix(backdoor_labels_lstm, pred_test)
    print(classification_report(backdoor_labels_lstm, pred_test))









def mlp():
    METRICS = [
    'acc',
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    f1_m,
    precision_m, 
    recall_m
    ]
    #compute class weights
    class_weights = class_weight.compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(y_train),
                                            y = y_train                                                    
                                        )
    model = keras.Sequential([
        keras.layers.Input(shape=(768,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    print(class_weights)
    class_weights = dict(zip(np.unique(y_train), class_weights))
    print(class_weights)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3) ,loss = 'binary_crossentropy', metrics=METRICS)
    return model.fit(X_train, y_train,epochs=100, batch_size=512, class_weight=class_weights, validation_data=(X_val, y_val), verbose=0)





def experimentsWithMultiplethreshold(test_pplm, test_clear):
    #average prediction probability on clear data
    thresholds = np.linspace(0, 0.99, 10)
    #tresholds = [0]
    average_prediction_clear = 0
    average_prediction_pplm = 0
    average_prediction_lstm = 0
    precisions = []
    recalls = []
    f1s = []
    i = 0
    prediction_clear_fix = model.predict(test_clear)
    #perform the experiment for each threshold
    for threshold in thresholds:
        prediction_clear = model.predict(test_clear)
        print("average probability on clear example ", np.average(prediction_clear))
        average_prediction_clear += np.average(prediction_clear)
        #average prediction probability on lstm
        prediction_lstm = model.predict(embeddings_lstm)
        print("average probability on lstm example", np.average(prediction_lstm))
        average_prediction_lstm += np.average(prediction_lstm)

        #average prediction probability on pplm
        prediction_pplm = model.predict(test_pplm)
        print("average probability on plm example", np.average(prediction_pplm))
        average_prediction_pplm += np.average(prediction_pplm)
        threshold = np.average(prediction_clear_fix) + threshold
        print("applied threshold: ", threshold)

        y_pred_bool_reduct = model.predict(X_test) >  threshold#np.average(prediction_clear_fix)+treshold
        y_pred_bool = model.predict(X_test) >= 0.5
        y_pred_bool_lstm = prediction_lstm >= 0.5
        # y_pred_bool = np.argmax(prediction_lstm, axis=1)
        y_pred_bool_lstm_reduct = prediction_lstm >  threshold#np.average(prediction_clear)+treshold

        recall_pplm = recall_score(y_test, y_pred_bool_reduct)

        precision_pplm = precision_score(y_test, y_pred_bool_reduct)

        f1_pplm = f1_score(y_test, y_pred_bool_reduct)
        # print(classification_report(y_test, y_pred_bool_reduct))
        precisions.append(precision_pplm)
        recalls.append(recall_pplm)
        f1s.append(f1_pplm)

        conf_matrix(y_test, y_pred_bool_reduct)
        conf_matrix(y_test, y_pred_bool)


        conf_matrix(backdoor_labels_lstm, y_pred_bool_lstm_reduct)
        conf_matrix(backdoor_labels_lstm, y_pred_bool_lstm)





if __name__ == "__main__":
    #main
    data_total_plm, data_total_lstm, data_clear, data_lstm, data_pplm= read_dataset()
    y = data_total_plm[0] 
    backdoor_labels = data_total_plm[1] 
    embeddings = data_total_plm[2]

    #get lstm generated data
    y_clear_lstm = data_total_lstm[0]
    backdoor_labels_clear_lstm = data_total_lstm[1]
    embeddings_clear_lstm = data_total_lstm[2]

    #get clear data
    y_clear = data_clear[0]
    backdoor_labels_clear = data_clear[1]
    embeddings_clear = data_clear[2]

    y_lstm = data_lstm[0]
    backdoor_labels_lstm = data_lstm[1]
    embeddings_lstm = data_lstm[2]

    y_pplm = data_pplm[0]
    backdoor_labels_pplm = data_pplm[1]
    embeddings_pplm = data_pplm[2]
    percentage_plot(backdoor_labels)
    #data for SVC classifier
    X_train, X_test, y_train, y_test = train_test_split(embeddings, backdoor_labels, test_size= 0.3, stratify=backdoor_labels)
    #make sure that both in train and test dataset the percentage of clear and backdoor data are the same as the original dataset


    #build and make experiment using support vector classifier
    svc()

    model = mlp()



    x_test_set = set(tuple(x) for x in X_test)
    test_pplm_set = set(tuple(x) for x in embeddings_pplm)
    test_pplm = np.array([x for x in x_test_set & test_pplm_set])

    test_clear_set = set(tuple(x) for x in embeddings_clear)
    test_clear = np.array([x for x in x_test_set & test_clear_set])

    print("dimension vector test clear: ", test_clear.shape)
    print("dimension vector test pplm: ", test_pplm.shape)

    #MLP classifier
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.3, stratify=y_train)
    experimentsWithMultiplethreshold(test_pplm, test_clear)








