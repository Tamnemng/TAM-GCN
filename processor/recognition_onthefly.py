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
    @staticmethod
    def get_parser(add_help=False):
        parent_parser = REC_Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='On-the-Fly ResNet + CTR-GCN Processor')

        parser.add_argument('--lr_scheduler', type=str, default='step',
                            help='LR scheduler: step or cosine')
        parser.add_argument('--warmup_epochs', type=int, default=0,
                            help='Number of warmup epochs for cosine scheduler')
        parser.add_argument('--label_smoothing', type=float, default=0.0,
                            help='Label smoothing factor for CrossEntropyLoss')
        parser.add_argument('--exp_type', type=str, default=None,
                            help='Experiment type for cross-attention ablation (normal, noise, ones, zeros, no_spatial)')
        parser.add_argument('--rgb_path', type=str, default=None,
                            help='Override the rgb_path defined in config for both train and test feeder args')
        return parser

    def load_model(self):
        # Override exp_type if provided via CLI
        if getattr(self.arg, 'exp_type', None) is not None:
            if 'resnet_args' not in self.arg.model_args:
                self.arg.model_args['resnet_args'] = {}
            self.arg.model_args['resnet_args']['exp_type'] = self.arg.exp_type

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
        
        # Label smoothing for better generalization
        label_smoothing = getattr(self.arg, 'label_smoothing', 0.0)
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def load_data(self):
        # Override rgb_path if provided via CLI
        if getattr(self.arg, 'rgb_path', None) is not None:
            if hasattr(self.arg, 'train_feeder_args') and isinstance(self.arg.train_feeder_args, dict):
                self.arg.train_feeder_args['rgb_path'] = self.arg.rgb_path
            if hasattr(self.arg, 'test_feeder_args') and isinstance(self.arg.test_feeder_args, dict):
                self.arg.test_feeder_args['rgb_path'] = self.arg.rgb_path
        
        # Superclass handles the data_path override and DataLoader creation
        super(REC_Processor_OnTheFly, self).load_data()

    def start(self):
        self.io.print_log(f'Parameters:\n{str(vars(self.arg))}')
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()

        if self.arg.phase == 'test':
            self.test()
            return

        best_acc = 0.0
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            self.epoch = epoch
            self.train()
            if epoch % self.arg.eval_interval == 0:
                acc = self.test()
                self.save_model(name=f'epoch{epoch+1}_model')
                if acc is not None and acc > best_acc:
                    best_acc = acc
                    self.save_model(name='best_model')
                    self.io.print_log(f'\t[*] New best accuracy: {best_acc*100:.2f}%')
        
    def adjust_learning_rate(self, epoch, step, base_lr):
        """Cosine annealing with warmup, or fallback to step decay."""
        lr_scheduler = getattr(self.arg, 'lr_scheduler', 'step')
        warmup_epochs = getattr(self.arg, 'warmup_epochs', 0)
        
        if lr_scheduler == 'cosine':
            if epoch < warmup_epochs:
                # Linear warmup
                lr = base_lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup_epochs) / max(1, self.arg.num_epoch - warmup_epochs)
                lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            # Original step decay
            lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _check_model_supports_labels(self):
        """Check once if the model's forward() accepts a 'labels' kwarg (V11+)."""
        if not hasattr(self, '_supports_labels'):
            import inspect
            # Handle DataParallel wrapper
            model = self.model.module if hasattr(self.model, 'module') else self.model
            fwd_params = inspect.signature(model.forward).parameters
            self._supports_labels = 'labels' in fwd_params
        return self._supports_labels

    def train(self):
        self.model.train()
        self.model.ctrgcn.eval() # Keep it in eval mode!
        self.adjust_learning_rate(self.epoch, self.arg.step, self.arg.base_lr)
        loader = self.data_loader['train']
        loss_value = []
        supports_labels = self._check_model_supports_labels()
        
        for data, rgb, label in loader:
            data = data.float().to(self.dev)
            rgb = rgb.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward — V11+ models may return (output, extra_loss) tuple
            if supports_labels:
                result = self.model(data, rgb, labels=label)
            else:
                result = self.model(data, rgb)
            
            if isinstance(result, tuple):
                output, extra_loss = result
            else:
                output = result
                extra_loss = 0.0

            loss = self.loss(output, label) + extra_loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.optimizer.param_groups[0]['lr'])
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

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

            # show top-k accuracy and return top-1
            for k in self.arg.show_topk:
                self.show_topk(k)
            
            # Return top-1 accuracy for best model tracking
            rank = self.result.argsort()
            hit_top1 = [l in rank[i, -1:] for i, l in enumerate(self.label)]
            return sum(hit_top1) * 1.0 / len(hit_top1)
        return None
