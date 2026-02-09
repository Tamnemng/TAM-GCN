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
        # Load model từ config
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args)
        
        # Định nghĩa Loss function
        # Không cần .cuda() ở đây, class cha sẽ tự chuyển sang GPU sau
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
            raise ValueError()

    def save_model(self, name='model'):
        """Save the model to the work directory."""
        filename = f'{name}.pt'
        self.io.save_model(self.model, filename)
        self.io.print_log(f'Model saved: {filename}')

    def adjust_learning_rate(self, epoch, step, base_lr):
        lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.model.train()
        self.adjust_learning_rate(self.epoch, self.arg.step, self.arg.base_lr)
        loader = self.data_loader['train']
        loss_value = []

        # Tqdm để hiện thanh tiến trình
        for data, label, _ in tqdm(loader, desc=f'Epoch {self.epoch+1}'):
            # SỬA LỖI: Dùng self.dev thay vì self.output_device
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            output = self.model(data)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.item())

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.io.print_log(f'\tTraining loss: {self.epoch_info["mean_loss"]:.4f}')

    def test(self):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        with torch.no_grad():
            for data, label, _ in tqdm(loader, desc='Evaluation'):
                # SỬA LỖI: Dùng self.dev thay vì self.output_device
                data = data.float().to(self.dev)
                label = label.long().to(self.dev)

                output = self.model(data)
                loss = self.loss(output, label)
                
                loss_value.append(loss.item())
                result_frag.append(output.data.cpu().numpy())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)
        
        predict_label = np.argmax(self.result, axis=1)
        acc = np.sum(predict_label == self.label) / len(self.label)
        if acc > self.meta_info['best_t1']:
            self.meta_info['best_t1'] = acc
            self.meta_info['is_best'] = True
        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['val_acc'] = acc
        self.io.print_log(f'\tEvaluation Acc: {acc:.2%}')

    def start(self):
        self.io.print_log(f'Parameters:\n{str(vars(self.arg))}')
        self.io.print_log(f'Work dir: {self.arg.work_dir}')
        
        self.load_model()
        self.load_weights()
        self.gpu() 
        self.load_data()
        self.load_optimizer()
        
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            self.epoch = epoch
            self.train()
            
            if epoch % self.arg.eval_interval == 0:
                self.test()
                # GỌI HÀM LƯU MODEL TẠI ĐÂY
                # save_model sẽ tự động lưu vào thư mục work_dir được định nghĩa trong file yaml
                self.save_model(name=f'epoch{epoch+1}_model') 
                
                # Nếu muốn lưu model tốt nhất (best model)
                if self.meta_info['is_best']:
                    self.save_model(name='best_model')
                    self.meta_info['is_best'] = False # Reset lại flag

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='ResNet Only Processor')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

        return parser