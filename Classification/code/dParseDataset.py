'''
Part of this code has inspired by the BERT fine-tuning tutorial by Chris McCormick 
(https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
'''

import pandas as pd

DATASET = {
            'test1':'../data/metaphor-src-test-1.csv', 
            'test2':'../data/metaphor-src-test-2.csv',
            'test3':'../data/metaphor-src-test-3.csv',
            'test4':'../data/metaphor-src-test-4.csv',
            'train_part1':'../data/metaphor-scm-train-part1.csv', 
            'dev_part1':'../data/metaphor-scm-dev-part1.csv',
            'test_part1':'../data/metaphor-scm-test-part1.csv'
        }

df_train = pd.read_csv(DATASET['train_part1'])
df_dev = pd.read_csv(DATASET['dev_part1'])
df_test = pd.read_csv(DATASET['test_part1']) 

xInput_train = df_train['Sentence'] + '<SEP>' + df_train['Source LM']
xLabels_train = df_train['Source CM']

xInput_dev = df_dev['Sentence'] + '<SEP>' + df_dev['Source LM']
xLabels_dev = df_dev['Source CM']

xInput_test = df_test['Sentence'] + '<SEP>' + df_test['Source LM']
xLabels_test = df_test['Source CM']

