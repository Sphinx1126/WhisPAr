# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 22:16:41 2023

@author: 28257
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import argparse
from dataset import WhisPArDataset
from models.WhisPAr import WhisPAr
import time
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import transformers
import torch.nn.functional as F

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=r'datasets/')
    parser.add_argument('--model_path', default=r'output/Model.pt')
    parser.add_argument('--output_path', default='output')
    parser.add_argument('--cls_num', type=int, default=6)
    parser.add_argument('--audio_len', default=30,type=int)
    parser.add_argument('--prompt_len', type=int, default=2)
    parser.add_argument('--bottleneck_dim', type=int, default=16)
    parser.add_argument('--adapt_list', type=list, default=None)
    parser.add_argument('--bs_eval', type=int, default=16)
    # parser.add_argument("--do_test", action='store_true', default=True)
    args = parser.parse_args()
    return args

def evaluate(args, model, dataloader):
    model.eval()
    device = args.device
    logger.info("Running evaluation")
    eval_loss = 0.0
    truth=[]
    pred=[]
    with torch.no_grad():
        for data in tqdm(dataloader):
            mel,label = data
            mel = mel.to(device)
            label = label.to(device)
            opts = model(mel,label)
            loss=opts[1]
            prob=opts[0]
            pred+=torch.argmax(prob,dim=1).tolist()
            truth+=label.tolist()
            eval_loss += loss
    acc=sum([truth[i]==pred[i] for i in range(len(truth))])/len(truth)
    return acc,eval_loss/len(dataloader),pred,truth

if __name__ == '__main__':
    args = set_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'test-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
    
    model = WhisPAr(cls_num=args.cls_num, audio_len=args.audio,
                    prompt_len=args.prompt_len,
                    bottleneck_dim=args.bottleneck_dim,adapt_list=args.adapt_list).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    
    test_dataset = WhisPArDataset(args.data_dir+'test.pkl')
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs_eval, shuffle=False)
    del test_dataset
    test_acc,test_loss,pred,truth=evaluate(args, model, test_dataloader)
    del test_dataloader
    logger.info('accuracy in Test Dataset is {}, loss is {}'.format(test_acc, test_loss.item()))
    
    df=pd.DataFrame()
    df['label']=truth
    df['pred']=pred
    df.to_csv(args.output_path+'/predict.csv')