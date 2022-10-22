import pandas as pd
import torch 
import numpy as np
# import plotting libraries
import matplotlib.pyplot as plt
from pathlib import Path
# import precision
from sklearn.metrics import classification_report, accuracy_score
# import gpu_config
from gpuConfig import device
import torch.nn as nn

softmax = nn.Softmax(dim=0)
sigmoid = nn.Sigmoid()

class Eval():
  def __init__(self, model, prediction_dataloader):
    # Put model in evaluation mode
    self.model = model
    self.model.eval()
    
    self.predictions , self.true_labels = list(), list()
    # Predict 
    for batch in prediction_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch
      
      with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, attention_mask=b_input_mask)
      logits = outputs[0]
      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      # Store predictions and true labels
      self.predictions.append(logits)
      self.true_labels.append(label_ids)

    print(' DONE.')
    print('******************************* Classification report: *******************************\n')

    self.y_true = list() 
    self.y_pred = list()
    for i in range(len(self.true_labels)):
      pred_labels_i = np.argmax(self.predictions[i], axis=1)
      self.y_true += self.true_labels[i].tolist()
      self.y_pred += pred_labels_i.tolist()
    
    print('Report via SKlearn:\n')
    print(classification_report(self.y_true, self.y_pred))

    print('Accuracy score:\n')
    print(accuracy_score(self.y_true, self.y_pred))


