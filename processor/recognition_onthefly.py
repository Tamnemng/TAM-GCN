import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math

from .recognition_rgb import REC_Processor

class REC_Processor_OnTheFly(REC_Processor):
    def load_model(self):
        # Load the combined ResNet + CTR-GCN Wrapper
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args['resnet_args']))
        
        # Load the authentic CTR-GCN model
        self.model.ctrgcn = self.io.load_model('models.ctrgcn.Model',
                                                **(self.arg.model_args['gcn_args']))
                                                
        # Load the frozen pre-trained weights for CTR-GCN
        if 'gcn_weights' in self.arg.model_args and self.arg.model_args['gcn_weights']:
            self.model.ctrgcn = self.io.load_weights(self.model.ctrgcn, self.arg.model_args['gcn_weights'],
                                                    self.arg.ignore_weights)
                                                    
        # Freeze CTR-GCN completely for on-the-fly feature extraction
        for param in self.model.ctrgcn.parameters():
            param.requires_grad = False
        self.model.ctrgcn.eval()
        
        self.loss = nn.CrossEntropyLoss()
        
    def train(self):
        self.model.train()
        self.model.ctrgcn.eval() # Keep it in eval mode!
        self.adjust_learning_rate(self.epoch, self.arg.step, self.arg.base_lr)
        loader = self.data_loader['train']
        loss_value = []
        
        for data, rgb, label in loader:
            data = data.float().to(self.dev)
            rgb = rgb.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data, rgb)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        self.model.ctrgcn.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, rgb, label in loader:
            data = data.float().to(self.dev)
            rgb = rgb.float().to(self.dev)
            label = label.long().to(self.dev)

            with torch.no_grad():
                output = self.model(data, rgb)
                
            result_frag.append(output.data.cpu().numpy())

            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
