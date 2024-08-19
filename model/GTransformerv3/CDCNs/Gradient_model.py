import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,out_channels,width,height,shifts):
        super().__init__()
        #The hyper parameters settting
        self.convs_list=nn.ModuleList()
        delta1=np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels,1)
        self.shifts =shifts
        
        self.scale=torch.nn.Parameter(torch.zeros(len(self.shifts)))
    
        #After Extraction, we analyze the outcome of the extraction.
        self.num_layer= 8
        self.width = width
        self.height = height
        self.area = width* height
        self.psi = nn.InstanceNorm2d(len(self.shifts))
        self.position_embeddings = nn.Parameter(torch.zeros(1,1,self.area))
        # self.layernorm1 = nn.LayerNorm(self.area)
        # self.layernorm2 = nn.LayerNorm(self.area)
        # self.layernorm3 = nn.LayerNorm(self.area)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.query_convs=nn.ModuleList()
        self.key_convs=nn.ModuleList()
        self.value_convs=nn.ModuleList()
        self.local_convs = nn.ModuleList()
        # self.hidden_channels = self.in_channels
            #The Process the of extraction of outcome
        self.kernel1 = torch.Tensor(w1).cuda()
        self.kernel2 = torch.Tensor(w2).cuda()
        self.kernel3 = torch.Tensor(w3).cuda()
        self.kernel4 = torch.Tensor(w4).cuda()
        self.kernel5 = torch.Tensor(w5).cuda()
        self.kernel6 = torch.Tensor(w6).cuda()
        self.kernel7 = torch.Tensor(w7).cuda()
        self.kernel8 = torch.Tensor(w8).cuda()
        self.kernel1 = self.kernel1.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel2 = self.kernel2.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel3 = self.kernel3.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel4 = self.kernel4.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel5 = self.kernel5.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel6 = self.kernel6.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel7 = self.kernel7.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel8 = self.kernel8.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.hidden_channels = self.in_channels//len(self.shifts)
        self.down_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,padding='same',bias=False)
        for i in range(len(self.shifts)):
            # kernel = max(self.shifts[i]-2,1)
            # self.down_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=kernel,stride=1,padding='same',bias=False))
            self.query_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False))
            self.local_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False))
            self.key_convs.append(nn.Conv2d(in_channels=self.in_channels*self.num_layer,out_channels=self.hidden_channels*self.num_layer,kernel_size=1,stride=1,bias=False))
            self.value_convs.append(nn.Conv2d(in_channels=self.in_channels*self.num_layer,out_channels=self.hidden_channels*self.num_layer,kernel_size=1,stride=1,bias=False)) 
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False),
                                      nn.BatchNorm2d(self.in_channels),
                                      nn.ReLU())
    def Extract_layer(self,cen,i=0):
        surrounds_keys = []
        surrounds_querys = []
        surrounds_values = []
        center_values = []
        # print(center.shape)
        # cen = self.down_conv(center)
        for i in range(len(self.shifts)):
            # cen = self.down_convs[i](center)
            surround1 = torch.nn.functional.conv2d(weight=self.kernel1, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround2 = torch.nn.functional.conv2d(weight=self.kernel2, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround3 = torch.nn.functional.conv2d(weight=self.kernel3, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround4 = torch.nn.functional.conv2d(weight=self.kernel4, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround5 = torch.nn.functional.conv2d(weight=self.kernel5, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround6 = torch.nn.functional.conv2d(weight=self.kernel6, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround7 = torch.nn.functional.conv2d(weight=self.kernel7, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround8 = torch.nn.functional.conv2d(weight=self.kernel8, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surrounds = torch.cat([surround1,surround2,surround3,surround4,surround5,surround6,surround7,surround8],1)
            surrounds_keys.append(self.key_convs[i](surrounds).flatten(2))
            surrounds_querys.append(self.query_convs[i](cen).flatten(2))
            center_values.append(self.local_convs[i](cen).flatten(2))
            surrounds_values.append(self.value_convs[i](surrounds).flatten(2))
        surrounds_keys = torch.stack(surrounds_keys,dim=1)
        surrounds_querys = torch.stack(surrounds_querys,dim=1)
        center_values = torch.stack(center_values,dim=1)
        surrounds_values = torch.stack(surrounds_values,dim=1)
        return surrounds_keys,surrounds_querys,surrounds_values,center_values
    def forward(self,cen):
        b,_,w,h= cen.shape
        deltas_keys,deltas_querys,deltas_values,local_values = self.Extract_layer(cen,i=0)
        deltas_keys = torch.nn.functional.normalize(deltas_keys,dim=-1).transpose(-2,-1)
        deltas_querys = torch.nn.functional.normalize(deltas_querys,dim=-1)
        weight_score = torch.matmul(deltas_querys,deltas_keys)
        weight_score_surround = self.softmax_layer(self.psi(weight_score))
        weight_score_1,weight_score_2,weight_score_3,weight_score_4,weight_score_5,weight_score_6,weight_score_7,weight_score_8 = torch.chunk(weight_score_surround,chunks=8,dim=-1)
        weight_score_center   = weight_score_1+weight_score_2+weight_score_3+weight_score_4+weight_score_5+weight_score_6+weight_score_7+weight_score_8
        out = torch.matmul(weight_score_surround,deltas_values) - torch.matmul(weight_score_center,local_values)
        out = out.view(b,self.in_channels,w,h)
        out = self.out_conv(out)
        return out