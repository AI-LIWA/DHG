from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.init as weight_init
import torch.nn.functional as F
import numpy as np
import copy
import scipy.io as scio
from tgcn import ConvTemporalGraphical
from ctrgcn import TCN_GCN_unit
#3 ->6
#from cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN
from graph import Graph
#from gIndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN
class Batch_norm_step(nn.Module):
    def __init__(self,  hidden_size,seq_len):
        super(Batch_norm_step, self).__init__()
        self.hidden_size = hidden_size
        
        self.max_time_step=seq_len
        self.bn = nn.BatchNorm1d(hidden_size) 

    def forward(self, x):
        x = x.permute(1,2,0)
        x = self.bn(x.clone())
        x = x.permute(2,0,1)
        return x

class Dropout_overtime(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, p=0.5,training=False):
    output = input.clone()
    noise = input.data.new(input.size(-2),input.size(-1))  #torch.ones_like(input[0])
    if training:            
      noise.bernoulli_(1 - p).div_(1 - p)
      noise = noise.unsqueeze(0).expand_as(input)
      output.mul_(noise)
    ctx.save_for_backward(noise)
    ctx.training=training
    return output
  @staticmethod
  def backward(ctx, grad_output):
    noise,=ctx.saved_tensors
    if ctx.training:
      return grad_output.mul(noise),None,None
    else:
      return grad_output,None,None
dropout_overtime=Dropout_overtime.apply

import argparse
import opts     
parser = argparse.ArgumentParser(description='pytorch action')
opts.train_opts(parser)
args = parser.parse_args()
MAG=args.MAG
U_bound=np.power(10,(np.log10(MAG)/args.seq_len))
U_lowbound=np.power(10,(np.log10(1.0/MAG)/args.seq_len))  
  
class stackedIndRNN_encoder(nn.Module):
    def __init__(self, input_size, outputclass,adaptive=True):
        super(stackedIndRNN_encoder, self).__init__()        
        hidden_size=args.hidden_size
        self.input_size=input_size
        self.graph = Graph(layout='hand',
                 strategy='spatial')
        #
        A = self.graph.A

      #  self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        in_channels=6
        base_channel=64
        self.relu = nn.ReLU(inplace=True)
        self.gcn=nn.ModuleList()
        self.gcn.append (TCN_GCN_unit(in_channels, base_channel, A, adaptive=adaptive))
        self.gcn.append(TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive))
        self.gcn.append(TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive))
        self.gcn.append(TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive))
        self.gcn.append(TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive))
        self.gcn.append(TCN_GCN_unit(base_channel*2, base_channel*4, A,stride=2, adaptive=adaptive))
        self.gcn.append(TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive))
        self.gcn.append(TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive))
       # self.gcn.append(ConvTemporalGraphical(256, 256,3
         #                               ))
      #  self.gcn.append(ConvTemporalGraphical(256, 256,3 ))

        self.l=nn.Parameter(torch.zeros(1))

        self.dr=nn.Dropout(0.5)
        self.data_bn = nn.BatchNorm1d(input_size *in_channels)

        self.BNs = nn.ModuleList()
     #   for x in range(args.num_layers):
      #      bn = Batch_norm_step(2*hidden_size,args.seq_len)
        #    self.BNs.append(bn)
        self.BNs.append(Batch_norm_step(2*hidden_size,args.seq_len))
        self.BNs.append(Batch_norm_step(2*hidden_size,args.seq_len))
        self.BNs.append(Batch_norm_step(2*hidden_size,args.seq_len))
        self.BNs.append(Batch_norm_step(2*hidden_size,args.seq_len))
        self.BNs.append(Batch_norm_step(2*hidden_size,args.seq_len))
        self.BNs.append(Batch_norm_step(2*hidden_size,args.seq_len))
        #  

        self.lastfc1 = nn.Linear(256, outputclass, bias=True)
        self.lastfc = nn.Linear(2*hidden_size, outputclass, bias=True)
        self.init_weights()
        self.soft=nn.Softmax(1)
        self.lastfc2 = nn.Linear(outputclass, outputclass, bias=True)
        self.conv1 = nn.Conv1d(in_channels=1024,out_channels = 66, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels=1024,out_channels = 66, kernel_size = 3)

    def init_weights(self):
      for name, param in self.named_parameters():
        if 'weight_hh' in name:
          param.data.uniform_(0,U_bound)          
        if 'RNNs.'+str(args.num_layers-1)+'.weight_hh' in name:
          param.data.uniform_(U_lowbound,U_bound)    
        if 'DIs' in name and 'weight' in name:
          param.data.uniform_(-args.ini_in2hid,args.ini_in2hid)               
        if 'bns' in name and 'weight' in name:
          param.data.fill_(1)      
        if 'bias' in name:
          param.data.fill_(0.0)
        if  'Conv2d' in name:
            param.data.uniform_(0,0.02)

    def forward(self, input):
        all_output = []
        rnnoutputs={}
        resoutputs={}
        hidden_x1={}
        hidden_x2={}
        cnnoutputs1={}
        cnnoutputs0={}


        seq_len, batch_size, indim,channel=input.size()
        input=input.permute(1,2,3,0).contiguous().view(batch_size,-1,seq_len)

        input=self.data_bn(input).view(batch_size,indim,channel,seq_len).permute(3,0,1,2).contiguous()
        input1= input.permute(1,3,0,2).contiguous()
        input=input.view(seq_len,batch_size,channel*indim)
        #

        rnnoutputs['rnnlayer0']=input
        cnnoutputs0['rnnlayer0']=input1

        for x in range(1,len(self.gcn)+1):


          cnnoutputs0['rnnlayer%d'%(x)]= self.gcn[x-1](cnnoutputs0['rnnlayer%d'%(x-1)])

        cout=cnnoutputs0['rnnlayer%d'%len(self.gcn)]
        cout = F.avg_pool2d(cout, cout.size()[2:])
        batch_size=cout.size()[0]
        cout=cout.view(batch_size,-1)
        output = self.lastfc1(cout)


        return output
class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x)


