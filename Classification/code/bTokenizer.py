'''
Part of this code has inspired by the BERT fine-tuning tutorial by Chris McCormick 
(https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
'''

from transformers import BertTokenizer
from dParseDataset import xInput_train, xLabels_train, xInput_dev, xLabels_dev, xInput_test, xLabels_test
import torch
from sklearn import preprocessing
import pandas as pd

le = preprocessing.LabelEncoder()

len_train = len(xLabels_train)
len_dev = len(xLabels_dev)
len_test = len(xLabels_test)

xAllLabels = le.fit_transform(pd.concat([xLabels_train,xLabels_dev,xLabels_test],axis=0,ignore_index=True))

xAllLabels = list(xAllLabels)
xLabels_train = xAllLabels[:len_train]
xLabels_dev = xAllLabels[len_train:len_train+len_dev]
xLabels_test = xAllLabels[len_train+len_dev:]

# Load the BERT tokenizer.
print('Loading bert tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

################### Tokenize dataset ################################

## ------------- TRAIN ------------- ##

xInputIds_train = []
xAttentionMasks_train = []

for each in xInput_train:
    encoded_dict_train = tokenizer.encode_plus(
                        str(each),                                   
                        add_special_tokens = True,                   
                        max_length = 128,                            
                        pad_to_max_length = True,
                        return_attention_mask = True,                
                        return_tensors = 'pt',                       
                   )
       
    xInputIds_train.append(encoded_dict_train['input_ids'])
    xAttentionMasks_train.append(encoded_dict_train['attention_mask'])

# Convert the lists into tensors
xInputIds_train = torch.cat(xInputIds_train,dim=0)
xAttentionMasks_train = torch.cat(xAttentionMasks_train,dim=0)
xLabels_train = torch.tensor(xLabels_train, dtype=torch.long)

## ------------- DEV ------------- ##

xInputIds_dev = []
xAttentionMasks_dev = []

for each in xInput_dev:
    encoded_dict_dev = tokenizer.encode_plus(
                        str(each),                                   
                        add_special_tokens = True,                   
                        max_length = 128,                            
                        pad_to_max_length = True,
                        return_attention_mask = True,                
                        return_tensors = 'pt',                       
                   )
       
    xInputIds_dev.append(encoded_dict_dev['input_ids'])
    xAttentionMasks_dev.append(encoded_dict_dev['attention_mask'])

# Convert the lists into tensors. 
xInputIds_dev = torch.cat(xInputIds_dev,dim=0)
xAttentionMasks_dev = torch.cat(xAttentionMasks_dev,dim=0)
xLabels_dev = torch.tensor(xLabels_dev, dtype=torch.long)

## ------------- TEST ------------- ##

xInputIds_test = []
xAttentionMasks_test = []

for each in xInput_test:
    encoded_dict_test = tokenizer.encode_plus(
                        str(each),                                   
                        add_special_tokens = True,                   
                        max_length = 128,                            
                        pad_to_max_length = True,
                        return_attention_mask = True,                
                        return_tensors = 'pt',                       
                   )
       
    xInputIds_test.append(encoded_dict_test['input_ids'])
    xAttentionMasks_test.append(encoded_dict_test['attention_mask'])

xInputIds_test = torch.cat(xInputIds_test,dim=0)
xAttentionMasks_test = torch.cat(xAttentionMasks_test,dim=0)
xLabels_test = torch.tensor(xLabels_test, dtype=torch.long)

