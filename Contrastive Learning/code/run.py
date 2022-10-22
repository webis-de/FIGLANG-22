'''
@All OS commands to run models for 20 iteraitons 
'''

import os

''' 
    run sBERT with Bert 
'''

for i in range(20):
    os.system("python3 sBertTrain.py 'bert-base-uncased' 0 0")

''' 
    run sBERT with DistilBert 
'''

for i in range(20):
    os.system("python3 sBertTrain.py 'distilbert-base-uncased' 0 0")