from __future__ import print_function
import sys
import argparse
import os
import time
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import opts

import Indrnn_action_network
import scipy.io as scio

from tqdm import tqdm

seed=100

a=np.arange(10)
b=np.reshape(a, (2,5))
c=np.tile(b,(1,1))


parser = argparse.ArgumentParser(description='pytorch action')
opts.train_opts(parser)
args = parser.parse_args()
print(args)
print('\n')


batch_size = args.batch_size
seq_len=args.seq_len
batch_sizet=64

outputclass= 14

indim=22
gradientclip_value=10
U_bound=Indrnn_action_network.U_bound


model = Indrnn_action_network.stackedIndRNN_encoder(indim, outputclass)
model.cuda()
criterion = nn.CrossEntropyLoss()

loss_fn = nn.NLLLoss()

#Adam with lr 2e-4 works fine.
learning_rate=args.lr
if args.use_weightdecay_nohiddenW:
  param_decay=[]
  param_nodecay=[]
  for name, param in model.named_parameters():
    if 'weight_hh' in name or 'bias' in name:
      param_nodecay.append(param)      
      #print('parameters no weight decay: ',name)          
    else:
      param_decay.append(param)      
      #print('parameters with weight decay: ',name)          

  if args.opti=='sgd':
    optimizer = torch.optim.SGD([
            {'params': param_nodecay},
            {'params': param_decay, 'weight_decay': args.decayfactor}
        ], lr=learning_rate,momentum=0.9,nesterov=True)   
  else:                
    optimizer = torch.optim.Adam([
            {'params': param_nodecay},
            {'params': param_decay, 'weight_decay': args.decayfactor}
        ], lr=learning_rate) 
else:  
  if args.opti=='sgd':   
    optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,nesterov=True)
  else:                      
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)


train_datasets='./class14/train_ntus'
test_dataset='./class14/test_ntus'


from data_reader_numpy_witheval import DataHandler_train
from data_reader_numpy_test import DataHandler as testDataHandler

dh_train = DataHandler_train(batch_size,seq_len)

dh_test= testDataHandler(batch_sizet,seq_len)

print('\n')

num_train_batches=int(np.ceil(dh_train.GetDatasetSize()/(batch_size+0.0)))

num_test_batches=int(np.ceil(dh_test.GetDatasetSize()/(batch_sizet+0.0)))

def clip_gradient(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip,clip)
        #print(p.size(),p.grad.data)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr     

def clip_weight(RNNmodel, clip):
    for name, param in RNNmodel.named_parameters():
      if 'weight_hh' in name:
        param.data.clamp_(-clip,clip)

def train(num_train_batches):
  model.train()
  tacc=0
  count=0
  start_time = time.time()
  loss_value = 0
  #pbar = tqdm(total = num_train_batches)
  for batchi in range(0,num_train_batches):
    inputs,targets=dh_train.GetBatch()
    inputs=inputs.transpose(1,0,2,3)

    #inputs=inputs[:,:,:,3:6]

    #cuda
    inputs=Variable(torch.from_numpy(inputs).cuda(), requires_grad=True)
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda(), requires_grad=False)

    # inputs=Variable(torch.from_numpy(inputs), requires_grad=True)
    # targets=Variable(torch.from_numpy(np.int64(targets)), requires_grad=False)

    model.zero_grad()
    if args.constrain_U:
      clip_weight(model,U_bound)
    output=model(inputs)
    loss = criterion(output, targets)

    pred = output.data.max(1)[1] # get the index of the max log-probability
    accuracy = pred.eq(targets.data).cpu().sum().numpy()/(0.0+targets.size(0))      
          
    loss.backward()
   # clip_gradient(model,gradientclip_value)
    optimizer.step()
    loss_value=loss_value+loss
    tacc=tacc+accuracy#loss.data.cpu().numpy()#accuracy
    count+=1
    #pbar.update(1)
    
  elapsed = time.time() - start_time
  #pbar.close()
  train_acc = tacc/(count+0.0) 
  print ("training accuracy: ", tacc/(count+0.0)  )
  print ("training loss: ", loss_value/(count+0.0)  )
  #print ('time per batch: ', elapsed/num_train_batches)
  
def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.train()       
def eval(dh,num_batches,use_bn_trainstat=False):
  model.eval()
  #pbar = tqdm(total=num_batches)
  if use_bn_trainstat:
    model.apply(set_bn_train)
  tacc=0
  count=0  
  start_time = time.time()
  while(1):  
    inputs,targets=dh.GetBatch()
    inputs=inputs.transpose(1,0,2,3)
   # inputs=inputs[:,:,:,3:6]

    inputs=Variable(torch.from_numpy(inputs).cuda())
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda())

    # inputs=Variable(torch.from_numpy(inputs))
    # targets=Variable(torch.from_numpy(np.int64(targets)))

    output=model(inputs)

    pred = output.data.max(1)[1] # get the index of the max log-probability
    accuracy = pred.eq(targets.data).cpu().sum().numpy()        
    tacc+=accuracy
    count+=1
    
    data_dict = {'total_ave_acc': total_ave_acc, 'testlabels': testlabels}
    with open('ST_gcn_pred_28.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    
    
    if count==num_batches*args.eval_fold:
      break
    #pbar.update(num_batches)
    
  #pbar.close()
  elapsed = time.time() - start_time
  print ("eval accuracy: ", tacc/(count*targets.data.size(0)+0.0)  )
  #print ('eval time per batch: ', elapsed/(count+0.0))
  return tacc/(count*targets.data.size(0)+0.0)
import pickle  
def test(dh,num_batches,use_bn_trainstat=False):
  model.eval()
  #print('aabbcc', num_batches)
  #pbar = tqdm(total=num_batches)  # num_batches=34
  if use_bn_trainstat:
    model.apply(set_bn_train)
  tacc=0
  tacc1=0
  count=0  
  start_time = time.time()
  total_testdata=dh_test.GetDatasetSize()  
  total_ave_acc=np.zeros((total_testdata,outputclass))
  testlabels=np.zeros((total_testdata))

  while(1):  
    inputs,targets,index=dh.GetBatch()
    inputs=inputs.transpose(1,0,2,3)
  #  inputs=inputs[:,:,:,3:6]
    testlabels[index]=targets
    inputs=Variable(torch.from_numpy(inputs).cuda())
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda())


    output=model(inputs)

    pred = output.data.max(1)[1] # get the index of the max log-probability
    #print('aaa', pred)
    #print('aaaac', pred.shape, output.shape)
    accuracy = pred.eq(targets.data).cpu().sum().numpy()

    total_ave_acc[index]+=output.data.cpu().numpy()
    
    tacc+=accuracy

    count+=1
    if count==num_batches*args.test_no:
      break    
  #total_ave_acc/=args.test_no
  top = np.argmax(total_ave_acc, axis=-1)
  
  np.save('stgcn_45_pred.npy',total_ave_acc)
  np.save('stgcn_45_label.npy',testlabels)
  eval_acc=np.mean(np.equal(top, testlabels))    
  elapsed = time.time() - start_time
  #pbar.update(1)
  print ("test accuracy: ", tacc/(count*targets.data.size(0)+0.0),'eval acc: ', eval_acc )
  #pbar.close()
  #print ('test time per batch: ', elapsed/(count+0.0))
  return eval_acc#tacc/(count*targets.data.size(0)+0.0)#, eval_acc/(total_testdata+0.0)

def save_train_log(train_log, epoch, train_acc, eval_acc):
    with open(train_log, 'a') as f:
        f.write(f'Epoch: {epoch} - Train Acc: {test_acc} -Eval Acc: {eval_acc}\n')

train_log = 'train.log'

lastacc=0
dispFreq=20
patience=0
reduced=1

print('num_train_batches = ' + str(num_train_batches))

print('\n')

#print('zzz',model)

for epoch in range(800):
  
  print('epoch = '+str(epoch))
  #print('a', indim, outputclass, U_bound)
  train(num_train_batches)
  test_acc=test(dh_test,num_test_batches)

  if (test_acc >lastacc):
    model_clone = copy.deepcopy(model.state_dict())   
    opti_clone = copy.deepcopy(optimizer.state_dict()) 

    lastacc=test_acc
    patience=0
  elif patience>int(args.pThre/reduced+0.5):
    reduced=reduced*2
    print ('learning rate',learning_rate)
    model.load_state_dict(model_clone)
    optimizer.load_state_dict(opti_clone)
    patience=0
    learning_rate=learning_rate*0.1
    adjust_learning_rate(optimizer,learning_rate)     
    if learning_rate<args.end_rate:
      break  

 
  else:
    patience+=1 
    #save_train_log()
    
test_acc=test(dh_test,num_test_batches)
#test_acc=test(dh_test,num_test_batches,True)

#torch.save(model, 'indrnn_action_model.pkl')
torch.save(model.state_dict(), 'indrnn_action_model_state.pkl')
torch.save(model_clone, 'best_action_model_state.pkl')
'''
save_name='indrnn_action_model' 
with open(save_name, 'wb') as f:
    torch.save(model, f)
'''
