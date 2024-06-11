

'''
Train and generate backdoor examples using lstm and pplm. The methods were retrieved from the following repository  https://github.com/lishaofeng/NLP_Backdoor
'''




import torch
torch.cuda.is_available()


from NLP_Backdoor.Toxic_Comment_Classification.PPLM import pplm_attack
from NLP_Backdoor.Toxic_Comment_Classification.PPLM import generator as generator_pplm

from NLP_Backdoor.Toxic_Comment_Classification.LSTM-BS import lstm_attack


import nltk
nltk.download('punkt')

sentences, labels = generator_pplm.prepare_data()
##### generate bad samples using pplm #######

pplm_attack.exp()


import numpy as np
a = np.array([1,2,3])
print(a)

##### generate bad samples using lstm #######
import nltk
nltk.download('punkt')

lstm_attack.exp()























































































































































import numpy as np
a = np.array([1,2,3])
print(a)















d

sd

re















from generator import *
opt = Config()
train(opt)

