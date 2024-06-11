# -*- coding: utf-8 -*-
"""

This file creates the embedding (using Bert model) for each sentence in the dataset. The methods used were retrieved from https://github.com/lishaofeng/NLP_Backdoor

"""



from NLP_Backdoor.Toxic_Comment_Classification.PPLM.generator import getDataloader
ids, segment, sentences, labels, backdoor = getDataloader("res/exp_beam_10_qsize_500",0.01, 10, 500)

import torch
import numpy as np

print((len(ids)))
print((len(segment)))
print((len(sentences)))
print((len(labels)))
print((len(backdoor)))

print(labels[0:20])
print(labels[len(labels)-1])

print(backdoor[0:20])
print(backdoor[len(labels)-1])

from transformers import BertModel
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True)
model.eval()

with torch.no_grad():
  embeddings = []
  i = 0
  for i in range(len(sentences)):
    tokens_tensor = torch.tensor([ids[i]])
    segments_tensor = torch.tensor([segment[i]])
    outputs = model(tokens_tensor, segments_tensor)

    # Evaluating the model will return a different number of objects based on
    # how it's  configured in the `from_pretrained` call earlier. In this case,
    # becase we set `output_hidden_states = True`, the third item will be the
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]
    token_vects = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vects, dim=0)
    embeddings.append(sentence_embedding)
    if i%25 == 0:
      print("fatte",i, "iterazioni")
    i = i+1
    #compute vector for each sentence
print(len(embeddings[0]))

len(embeddings)

#save vector in csv file
import pandas as pd
#embeddings = embeddings.numpy()
df = pd.DataFrame({"name":sentences, "label":labels, "backdoor": backdoor, "embedding":embeddings})
df.to_csv("data/dataset_summary_added_backdoor_data.csv", index=False)

print(sentences[len(sentences)-1])

df = pd.read_csv("data/dataset_summary.csv")

test = df['embedding']
test = test.replace('tensor', "")
test = test.replace('[', '')
test = test.replace(']', '')
test = test.replace('(', '')
test = test.replace(')', '')
print(test)
print(np.fromstring(test, spe=','))