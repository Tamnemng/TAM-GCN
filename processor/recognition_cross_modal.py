import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .processor import Processor
from torchlight import str2bool

class REC_Processor(Processor):
    def load_model(self):
        # Load model from config
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError("Unsupported optimizer")

    def adjust_learning_rate(self, epoch):
        lr = self.arg.base_lr * (
            self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self):
        epoch = self.meta_info['epoch']
        self.model.train()
        self.io.print_log('Training epoch: {}'.format(epoch + 1))
        lr = self.adjust_learning_rate(epoch)
        self.io.print_log('\tLearning rate: {}'.format(lr))
        
        loader = self.data_loader['train']
        loss_value = []
        for batch_idx, (data, label, index) in enumerate(tqdm(loader)):
            data_ske = data[0].float().to(self.dev)
            data_rgb = data[1].float().to(self.dev)
            label = label.long().to(self.dev)
            
            output = self.model(data_ske, data_rgb)
            loss = self.loss(output, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())

        self.io.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['ls_cls'] = np.mean(loss_value)

    def test(self, loader_name=['test']):
        epoch = self.meta_info['epoch']
        self.model.eval()
        self.io.print_log('Eval epoch: {}'.format(epoch + 1))
        
        for ln in loader_name:
            loss_value = []
            score_frag = []
            true_labels = []
            process = tqdm(self.data_loader[ln])
            
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data_ske = data[0].float().to(self.dev)
                    data_rgb = data[1].float().to(self.dev)
                    label = label.long().to(self.dev)
                    
                    output = self.model(data_ske, data_rgb)
                    loss = self.loss(output, label)
                    
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    true_labels.extend(label.cpu().numpy())
            
            score = np.concatenate(score_frag)
            predict_labels = np.argmax(score, axis=1)
            
            true_labels = np.array(true_labels)
            correct_predictions = np.sum(predict_labels == true_labels)
            accuracy = correct_predictions / len(true_labels)
            
            self.io.print_log('Accuracy: {:.2f}%'.format(100 * accuracy))
            self.epoch_info['ls_cls'] = 100 * accuracy

    def load_weights(self):
        if self.arg.weights:
            self.io.print_log(f'Loading CTR-GCN structural weights from {self.arg.weights}')
            weights = torch.load(self.arg.weights)
            try:
                # the model name in our new ResNet_GCN_Attention is self.gcn
                if hasattr(self.model, 'module'):
                    self.model.module.gcn.load_state_dict(weights) 
                else:
                    self.model.gcn.load_state_dict(weights)
                self.io.print_log("Load CTR-GCN weights success!")
            except Exception as e:
                self.io.print_log(f"Warning load GCN weights: {e}")

    def start(self):
        # Allow parent to set up everything
        super().start()

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Cross Modal Processor')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='Optimizer epoch to reduce learning rate')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
        parser.add_argument('--optimizer', default='SGD', help='Type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='Use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
        return parser
