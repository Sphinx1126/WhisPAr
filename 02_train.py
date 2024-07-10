# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:36:36 2023

@author: 28257
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import argparse
from dataset import WhisPArDataset
from models.PERTA import PERTA
from models.RelaGraph import KGEModel
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
    parser.add_argument('--output_path', default='output')
    parser.add_argument('--cls_num', type=int, default=6)
    parser.add_argument('--audio_len', default=30,type=int)
    parser.add_argument('--prompt_len', type=int, default=2)
    parser.add_argument('--bottleneck_dim', type=int, default=16)
    parser.add_argument('--adapt_list', type=list, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs_train', type=int, default=16)
    parser.add_argument('--dev_size', type=int, default=10)
    parser.add_argument('--bs_eval', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument("--save_step", type=int, default=6000)
    parser.add_argument("--eval_step", type=int, default=600)
    parser.add_argument("--do_train", action='store_true', default=True)
    # parser.add_argument("--do_test", action='store_true', default=True)
    args = parser.parse_args([])
    return args

def train(model, train_loader, dev_dataloader, optimizer, scheduler, args):
    model.train()
    logger.info("start training")
    device = args.device
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx + 1
            mel,label = data
            mel = mel.to(device)
            label = label.to(device)
            
            model.train()
            opts = model(mel,label)
            loss=opts[1]

            if step % args.eval_step == 0:
                dev_acc,dev_loss,_,_ = evaluate(args, model, dev_dataloader)
                writer.add_scalar('loss', dev_loss, step)
                logger.info('accuracy at step {} is {}, loss is {}'.format(step, dev_acc, dev_loss.item()))
                model.train()

            if step % args.save_step == 0:
                logger.info('saving checkpoint at step {}'.format(step))
                save_path = join(args.output_path, 'checkpoint-{}.pt'.format(step))
                torch.save(model.state_dict(), save_path)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    logger.info('saving model')
    save_path = join(args.output_path, 'Model.pt')
    torch.save(model.state_dict(), save_path)

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
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
        
    model = WhisPAr(cls_num=args.cls_num, audio_len=args.audio,
                    prompt_len=args.prompt_len,
                    bottleneck_dim=args.bottleneck_dim,adapt_list=args.adapt_list).to(args.device)
    
    train_dataset = WhisPArDataset(args.data_dir+'train.pkl')
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs_train, shuffle=True)
    del train_dataset
    val_dataset = WhisPArDataset(args.data_dir+'valid.pkl')
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs_eval, shuffle=False)
    del val_dataset
    
    t_total = len(train_dataloader) * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, args)
    logger.info('Training Done.')