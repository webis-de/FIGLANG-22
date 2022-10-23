'''
Part of this code has inspired by the BERT fine-tuning tutorial by Chris McCormick 
(https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
'''

# import libraries
from operator import attrgetter
import torch
import logging
# import torch specific libraries
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# import transformers specific libraries
from transformers import DistilBertForSequenceClassification, AdamW, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
# my own modules
from gpuConfig import device
from helper import flat_accuracy, format_time
# import basic modules 
import time
import random
import numpy as np
# import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import wandb

logging.basicConfig(level=logging.INFO)

class WoKDistBert():
    def __init__(self, batch_size, lr, eps, epochs, seed_val, __model): 
        self.batch_size = batch_size 
        self.lr = lr 
        self.eps = eps 
        self.epochs = epochs 
        self.seed_val = seed_val
        self.__model = __model
        
        if self.__model == 'distilbert':
            from dTokenizer import xInputIds_train, xAttentionMasks_train, xLabels_train, xInputIds_dev, xAttentionMasks_dev, xLabels_dev, xInputIds_test, xAttentionMasks_test, xLabels_test
            
            # Combine the training inputs into a TensorDataset.
            dataset_train = TensorDataset(xInputIds_train, xAttentionMasks_train, xLabels_train)
            dataset_dev = TensorDataset(xInputIds_dev, xAttentionMasks_dev, xLabels_dev)
            dataset_test = TensorDataset(xInputIds_test, xAttentionMasks_test, xLabels_test) 

            self.train_dataset, self.val_dataset, self.test_dataset = dataset_train, dataset_dev, dataset_test
            print('{:>5,} training samples'.format(len(self.train_dataset)))
            print('{:>5,} validation samples'.format(len(self.val_dataset)))
            print('{:>5,} test samples'.format(len(self.test_dataset)))
 
            self.train_dataloader = DataLoader(
                self.train_dataset,                                  # training samples.
                sampler = RandomSampler(self.train_dataset),         # select batches randomly
                batch_size = self.batch_size,                        # trains with this batch size.
            )
            
            self.validation_dataloader = DataLoader(
                self.val_dataset,                                    # validation samples.
                sampler = SequentialSampler(self.val_dataset),       # select batches sequentially.
                batch_size = self.batch_size                         # evaluate with this batch size.
            )

            prediction_sampler = SequentialSampler(self.test_dataset)
            self.prediction_dataloader = DataLoader(self.test_dataset, sampler=prediction_sampler, batch_size=self.batch_size)

            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',                              
                num_labels = 138,                                          
                output_attentions = False,                              
                output_hidden_states = False,                           
                return_dict=False
            )
        else:
            from bTokenizer import xInputIds_train, xAttentionMasks_train, xLabels_train, xInputIds_dev, xAttentionMasks_dev, xLabels_dev, xInputIds_test, xAttentionMasks_test, xLabels_test
            
            # Combine the training inputs into a TensorDataset.
            dataset_train = TensorDataset(xInputIds_train, xAttentionMasks_train, xLabels_train)
            dataset_dev = TensorDataset(xInputIds_dev, xAttentionMasks_dev, xLabels_dev)
            dataset_test = TensorDataset(xInputIds_test, xAttentionMasks_test, xLabels_test) 

            self.train_dataset, self.val_dataset, self.test_dataset = dataset_train, dataset_dev, dataset_test
            print('{:>5,} training samples'.format(len(self.train_dataset)))
            print('{:>5,} validation samples'.format(len(self.val_dataset)))
            print('{:>5,} test samples'.format(len(self.test_dataset)))

            self.train_dataloader = DataLoader(
                self.train_dataset,                                  # training samples.
                sampler = RandomSampler(self.train_dataset),         # select batches randomly
                batch_size = self.batch_size,                        # trains with this batch size.
            )

            self.validation_dataloader = DataLoader(
                self.val_dataset,                                    # validation samples.
                sampler = SequentialSampler(self.val_dataset),       # select batches sequentially.
                batch_size = self.batch_size                         # evaluate with this batch size.
            )

            prediction_sampler = SequentialSampler(self.test_dataset)
            self.prediction_dataloader = DataLoader(self.test_dataset, sampler=prediction_sampler, batch_size=self.batch_size)

            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',                             
                num_labels = 138,                                           
                output_attentions = False,                             
                output_hidden_states = False,                          
                return_dict=False
            )
            
        # Tell pytorch to run this model on the GPU.
        
        # model.cuda()
        
        self.optimizer = AdamW(self.model.parameters(),
                        lr = self.lr, 
                        eps = self.eps 
                        )

        self.total_steps = len(self.train_dataloader) * self.epochs
        
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, 
                                                    num_training_steps = self.total_steps)                                            
        
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

        self.training_stats = []
        
        self.total_t0 = time.time()

    def return_prediction_set(self):
        return self.prediction_dataloader

    def return_validation_set(self):
        return self.validation_dataloader

    def train(self):
        for epoch_i in range(0, self.epochs):

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')
            # Measure how long the training epoch takes.
            t0 = time.time()
            # Reset the total loss for this epoch.
            total_train_loss = 0
            # Put the model into training mode. 
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                
                if step % 60 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('Batch {:>5,}  of  {:>5,}. Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
        
                b_input_ids = batch[0].to(device); print('INPUT SHAPE: ',b_input_ids.shape)
                b_input_mask = batch[1].to(device); print('INPUT MASK SHAPE: ',b_input_mask.shape)
                b_labels = batch[2].to(device); print('TARGET SHAPE: ',b_labels.shape)
                
                self.model.zero_grad()        
                # Perform a forward pass 
                loss, logits = self.model(b_input_ids, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
                
                total_train_loss += loss.item()
                print('Current Loss', loss.item())
                
                # Perform a backward pass 
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                # Update the learning rate.
                self.scheduler.step()
            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_dataloader)            
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            print("")
            print("Average training loss: {0:.2f}".format(avg_train_loss))
            print("Training epoch took: {:}".format(training_time))

            train_metrics = {"training_loss": avg_train_loss}
            wandb.log(train_metrics)

            self.val()
        
        return self.model

    def val(self):
        print("")
        print("Running Validation...")
        t0 = time.time()
        self.model.eval()
        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in self.validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():        
                (loss, logits) = self.model(b_input_ids,  
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            # Accumulate the validation loss.
            total_eval_loss += loss.item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(self.validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        val_metrics = {"validation_accuracy": avg_val_accuracy,
               "validation_loss": avg_val_loss}
        wandb.log(val_metrics)


