'''
@All OS commands to run models for 20 iteraitons 
'''

import os

for i in range(20):
    os.system("python3 dMain.py 32 5e-5 1e-8 6 42 'bert'")