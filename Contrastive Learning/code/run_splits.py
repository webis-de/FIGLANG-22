'''
@All OS commands to run models for 5 iteraitons 
'''

import os

''' 
    run sBERT with Bert 
'''

for split in range(1,5):
    for i in range(5):
        os.system("python3 sBertTrain.py 'bert-base-uncased' {} {}".format(split,i+1))

''' 
    run sBERT with DistilBert 
'''

for split in range(1,5):
    for i in range(5):
        os.system("python3 sBertTrain.py 'distilbert-base-uncased' {} {}".format(split,i+1))
    
    
