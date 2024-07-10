# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:10:08 2023

@author: 28257
"""

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
import whisper

from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import torch.nn.functional as F
from loguru import logger

class Adapter(nn.Module):
    def __init__(self,input_dim,bottleneck_dim):
        super(Adapter, self).__init__()
        self.bottleneck_dim=bottleneck_dim
        self.down_project=nn.Linear(input_dim, self.bottleneck_dim)
        self.nonlinearity=nn.ReLU()
        self.up_project=nn.Linear(self.bottleneck_dim, input_dim)
        
        torch.nn.init.normal_(self.down_project.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.up_project.weight, mean=0.0, std=0.01)
    
    def forward(self,x):
        opt=self.down_project(x)
        opt=self.nonlinearity(opt)
        opt=self.up_project(opt)
        opt+=x
        return opt

class AdaptedAttentionBlock(nn.Module):
    def __init__(self,encoder_blcok,
                 input_dim,bottleneck_dim):
        super(AdaptedAttentionBlock, self).__init__()
        self.attn=encoder_blcok.attn
        self.attn_adapter=Adapter(input_dim,bottleneck_dim)
        self.attn_ln=encoder_blcok.attn_ln
        self.cross_attn =encoder_blcok.cross_attn
        self.cross_attn_adapter=Adapter(input_dim,bottleneck_dim) if self.cross_attn is not None else None
        self.cross_attn_ln =encoder_blcok.cross_attn_ln
        self.mlp =encoder_blcok.mlp
        self.mlp_adapter=Adapter(input_dim,bottleneck_dim)
        self.mlp_ln =encoder_blcok.mlp_ln
    
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn_adapter(self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0])
        if self.cross_attn:
            x = x + self.cross_attn_adapter(self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0])
        x = x + self.mlp_adapter(self.mlp(self.mlp_ln(x)))
        return x


class AdaptedEncoder(nn.Module):
    def __init__(self,encoder_blocks,prompt_len,
                 input_dim,bottleneck_dim,adapt_list=None):
        super(AdaptedEncoder, self).__init__()
        self.dim=input_dim
        self.block_num=len(encoder_blocks)
        self.prompt_len=prompt_len
        self.adapt_list=range(self.block_num) if adapt_list is None else adapt_list
        self.adpated_blocks=encoder_blocks
        for i in range(self.block_num):
            if i in self.adapt_list:
                self.adpated_blocks[i]=AdaptedAttentionBlock(encoder_blocks[i],
                                                             input_dim,bottleneck_dim)
            for param in self.adpated_blocks[i].attn_ln.parameters():
                param.requires_grad=True
            for param in self.adpated_blocks[i].mlp_ln.parameters():
                param.requires_grad=True
            if self.adpated_blocks[i].cross_attn_ln is not None:
                for param in self.adpated_blocks[i].cross_attn_ln.parameters():
                    param.requires_grad=True
        self.prompt=nn.Parameter(torch.zeros(self.prompt_len,self.dim*self.block_num), requires_grad=True)
    def forward(self, x: Tensor):
        bs=x.size(0)
        for i in range(self.block_num):
            prompt=self.prompt[:,i*self.dim:(i+1)*self.dim].unsqueeze(0).expand(bs,self.prompt_len,self.dim)
            x=torch.cat([x,prompt],dim=1)
            x=self.adpated_blocks[i](x)[:,:-self.prompt_len,:]
        return x
            
        

class WhisPAr(nn.Module):
    def __init__(self,cls_num,audio_len,
                 prompt_len,
                 bottleneck_dim,adapt_list=None):
        super(WhisPAr, self).__init__()
        self.cls_num=cls_num
        self.prompt_len=prompt_len
        self.bottleneck_dim=bottleneck_dim
        
        model=whisper.load_model("base")
        self.dim=model.dims.n_audio_state
        
        whisper_encoder=model.encoder
        for param in whisper_encoder.parameters():
            param.requires_grad=False
        
        self.conv1=whisper_encoder.conv1
        self.conv2=whisper_encoder.conv2
        self.position_embed=whisper_encoder.positional_embedding[:audio_len*50,:]
        
        self.cls_embed=nn.Parameter(torch.zeros(1,self.dim), requires_grad=True)
        
        self.encoder_blocks=AdaptedEncoder(whisper_encoder.blocks,
                                           self.prompt_len, 
                                           self.dim, bottleneck_dim,adapt_list)
        self.ln_post=whisper_encoder.ln_post
        for param in self.ln_post.parameters():
            param.requires_grad=True
        
        self.full_connect=nn.Linear(self.dim, self.cls_num)
        self.softmax=nn.Softmax(dim=1)
        self.loss_fn=nn.CrossEntropyLoss()
    
    
    def forward(self,x,labels=None):
        bs=x.size(0)
        
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + self.position_embed).to(x.dtype)
        cls_embed=self.cls_embed.unsqueeze(0).expand(bs,1,self.dim)
        x=torch.cat([cls_embed,x],dim=1)
        x = self.encoder_blocks(x)
        x = self.ln_post(x)
        
        logits=self.full_connect(x[:,0,:])
        prob=self.softmax(logits)
        if labels is not None:
            loss=self.loss_fn(prob,labels)
            return (prob,loss)
        return prob