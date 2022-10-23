'''
Part of this code has inspired by the BERT fine-tuning tutorial by Chris McCormick 
(https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
'''

# import tensorflow as tf
import torch

# Get the GPU device name. 
# device_name = tf.test.gpu_device_name()
# print(device_name)
# The device name should look like the following:
'''
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
'''

# # If there's a GPU available...
# if torch.cuda.is_available():    

#     # Tell PyTorch to use the GPU.    
#     device = torch.device("cuda")

#     print('There are %d GPU(s) available.' % torch.cuda.device_count())

#     print('We will use the GPU:', torch.cuda.get_device_name(0))

# # If not...
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")

device = torch.device("cpu")
