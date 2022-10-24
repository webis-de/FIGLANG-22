'''
Implementation based on Sentence-Transformers library by Nils Reimers 
(https://www.sbert.net/docs/training/overview.html)
'''

from ast import Raise
import pandas as pd
import csv
import logging
import os
import sys
from datetime import datetime
from zipfile import ZipFile
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from Evaluator import RankingEvaluator 
import wandb
from transformers import TrainingArguments, Trainer
import argparse

wandb.init(project="your_project", entity="your_name")

model_name = 'bert-base-uncased'

def train_model_single_task(model_name,
                task='scm',
                eval_subset='test',
                output_path='model_config/',
                num_epochs=3, 
                train_batch_size=32, 
                model_suffix='', 
                data_file_suffix='', 
                max_seq_length=128, 
                add_special_token=False, 
                loss='MultipleNegativesRankingLoss', 
                add_metaphor = True,
                test_split=0,
                iterations=0,
                add_lexical_trigger = False,
                sentence_transformer=False):

    # Configure sentence transformers for training and train on the provided dataset
    output_path = output_path+model_name+ "-" + model_suffix + "-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    TRAIN_PATH = '../data/metaphor-scm-train-part1.csv'
    DEV_PATH = '../data/metaphor-scm-dev-part1.csv'
    TEST_PATH = '../data/metaphor-scm-test-part1.csv'
    
    '''
    For the data splits  
    '''
    if test_split == 1:
        TEST_PATH = '../data/metaphor-src-test-1.csv'
        print('Testing on split 1')
    if test_split == 2:
        TEST_PATH = '../data/metaphor-src-test-2.csv'
        print('Testing on split 2')
    if test_split == 3:    
        TEST_PATH = '../data/metaphor-src-test-3.csv'
        print('Testing on split 3')
    if test_split == 4:    
        TEST_PATH = '../data/metaphor-src-test-4.csv'
        print('Testing on split 4')

    print("Using {} as encoder".format(model_name))

    if sentence_transformer:
        word_embedding_model = SentenceTransformer(model_name)
        word_embedding_model.max_seq_length = max_seq_length
        
        if add_special_token:
            word_embedding_model.tokenizer.add_tokens(['<SEP>'], special_tokens=True)
            word_embedding_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    else:
        word_embedding_model = models.Transformer(model_name) 
        word_embedding_model.max_seq_length = max_seq_length
    
        if add_special_token:
            word_embedding_model.tokenizer.add_tokens(['<SEP>'], special_tokens=True)
            word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_examples = []
    with open(TRAIN_PATH, encoding="utf-8") as fIn:
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if loss=='MultipleNegativesRankingLoss':
                if add_lexical_trigger == False:
                    train_examples.append(InputExample(texts=[row['Sentence'], row['Source CM']])) if add_metaphor == False else train_examples.append(InputExample(texts=[row['Sentence']+'<SEP>'+row['Source LM']+'<SEP>'+row['Target CM'], row['Source CM']]))
                else:
                    train_examples.append(InputExample(texts=[row['Sentence']+'<SEP>'+row['Lexical Trigger'], row['Source CM']])) if add_metaphor == False else train_examples.append(InputExample(texts=[row['Sentence']+'<SEP>'+row['Source LM']+'<SEP>'+row['Lexical Trigger'], row['Source CM']]))
            
    if loss == 'MultipleNegativesRankingLoss':
        # Special data loader that avoid duplicates within a batch
        train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=train_batch_size)
        # Training loss
        train_loss = losses.MultipleNegativesRankingLoss(model)
    
    if eval_subset=='dev': 
        var = pd.read_csv(DEV_PATH)
    else:
        var = pd.read_csv(TEST_PATH)
    
    if add_lexical_trigger == False:
        sentences = var['Sentence'] if add_metaphor == False else var['Sentence'] + '<SEP>' + var['Source LM']+'<SEP>'+row['Target CM']
        pos = var['Source CM']
    else: 
        sentences = var['Sentence'] + '<SEP>' + var['Lexical Trigger'] if add_metaphor == False else var['Sentence'] + '<SEP>' + var['Source LM'] + '<SEP>' + var['Lexical Trigger']
        pos = var['Source CM']

    assert len(sentences) == len(pos)

    anchors_with_ground_truth_candidates = dict(zip(list(sentences), list(pos)))

    evaluator = RankingEvaluator(anchors_with_ground_truth_candidates, 
                                task,
                                loss,
                                show_progress_bar=False,
                                name=eval_subset,
                                split = test_split,
                                iterations = 0 
                                )

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of train data

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              evaluator=evaluator,
              evaluation_steps=100,
              warmup_steps=warmup_steps,
              optimizer_params={'lr': 5e-5},
              show_progress_bar=True,
              output_path=output_path)

parser = argparse.ArgumentParser(description="Split entry")
parser.add_argument("model_name", type=str)
parser.add_argument("split", type=int)
parser.add_argument("itr", type=int)
args = parser.parse_args()

train_model_single_task(model_name=args.model_name, 
            task= 'scm', 
            eval_subset='test',
            num_epochs=6, 
            train_batch_size=32, 
            model_suffix='', 
            data_file_suffix='', 
            max_seq_length=128, 
            add_special_token=False, 
            loss='MultipleNegativesRankingLoss', 
            add_metaphor = True,
            test_split = args.split,
            iterations = args.itr, 
            add_lexical_trigger = False,
            sentence_transformer=False)


