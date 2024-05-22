# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:59:33 2023

@author: 28257
"""

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from os.path import join
from loguru import logger
import glob
import whisper

class WhisPArDataset(Dataset):
    def __init__(self,data_dir):
        with open(data_dir,'rb+') as fp:
            mel_list,label_list=pickle.load(fp)
        
        self.mel_list=mel_list
        self.label_list=[torch.tensor(label,dtype=torch.long) for label in label_list]
        
    def __len__(self) -> int:
        return len(self.label_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        mel = self.mel_list[index]
        label = self.label_list[index]
        return mel,label