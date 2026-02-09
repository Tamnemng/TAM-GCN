import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .processor import Processor # Kế thừa từ class Processor gốc

class REC_Processor(Processor):
    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        loss_value = []
        for batch_idx, (data, label, index) in enumerate(tqdm(loader)):
            data_ske = data[0].float().cuda(self.output_device)
            data_rgb = data[1].float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)
            output = self.model(data_ske, data_rgb)
            loss = self.loss(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())

        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        
        if save_model:
            torch.save(self.model.state_dict(), 
                       self.arg.model_saved_name + '-' + str(epoch) + '.pt')

    def test(self, epoch, save_score=False, loader_name=['test']):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        
        for ln in loader_name:
            loss_value = []
            score_frag = []
            process = tqdm(self.data_loader[ln])
            
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data_ske = data[0].float().cuda(self.output_device)
                    data_rgb = data[1].float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    
                    output = self.model(data_ske, data_rgb)
                    loss = self.loss(output, label)
                    
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
            
            score = np.concatenate(score_frag)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            self.print_log('Accuracy: {:.2f}%'.format(100 * accuracy))
    def start(self):
        if self.arg.weights:
            self.print_log(f'Loading weights from {self.arg.weights}')
            weights = torch.load(self.arg.weights)
            try:
                self.model.stgcn.load_state_dict(weights) 
                print("Load CTR-GCN weights success!")
            except Exception as e:
                print(f"Warning load weights: {e}")
        super().start()