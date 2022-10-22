from dBertInputEncoder import WoKDistBert
from dEval import Eval
import argparse
import wandb

'''
Optimized Params:
learning_rate = 5e-5
batch_size = 32
num_train_epochs = 6.0
'''

wandb.init(project="my_project")

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("batch_size", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("eps", type=float)
parser.add_argument("epochs", type=int)
parser.add_argument("seed_val", type=int)
parser.add_argument("__model", type=str)
args = parser.parse_args()

actvmodl = WoKDistBert(args.batch_size, args.lr, args.eps, args.epochs, args.seed_val, args.__model)

pred = actvmodl.return_prediction_set()
modl = actvmodl.train()
Eval(modl,pred)


