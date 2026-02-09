import torch
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, num_class=10, pretrained=True, **kwargs):
        super(Model, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)

    def forward(self, x):
        return self.model(x)