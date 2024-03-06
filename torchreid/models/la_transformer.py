
import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


__all__ = [
    'la_transformer'
]


class InitPretrainedWeights:

    def __int__(self):
        pass

    def __call__(self, model, pretrained_model=None, *args, **kwargs):
        if pretrained_model is None:
            pretrained_model = '/home/nwq/open-source/weight/LA-Transformer-net_best.pth'
        print(f"pretrained_model: {pretrained_model}")
        model.load_state_dict(torch.load(pretrained_model), strict=False)


# weights initialization
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class LATransformer(nn.Module):
    def __init__(self, model, lmbd, num_classes, loss, test_only):
        super(LATransformer, self).__init__()

        self.class_num = num_classes
        self.loss = loss
        self.test_only = test_only

        self.part = 14  # We cut the pool5 to sqrt(N) parts
        self.num_blocks = 12
        self.model = model
        self.model.head.requires_grad_ = False
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 768))
        self.dropout = nn.Dropout(p=0.5)
        self.lmbd = lmbd

        if not self.test_only:
            for i in range(self.part):
                name = 'classifier'+str(i)
                setattr(self, name, ClassBlock(768, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):

        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.pos_embed)

        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)
        x = self.model.norm(x)

        # extract the cls token
        cls_token_out = x[:, 0].unsqueeze(1)

        # Average pool
        x = self.avgpool(x[:, 1:])

        if not self.training:
            return x[:, 1, :]

        if self.loss == 'softmax':
            # Add global cls token to each local token4
            predict = [None] * self.part
            for i in range(self.part):
                # out = torch.mul(x[:, i, :], self.lmbd)
                # x[:, i, :] = torch.div(torch.add(cls_token_out.squeeze(), out), 1 + self.lmbd)

                # locally aware network
                name = 'classifier' + str(i)
                classifier = getattr(self, name)
                predict[i] = classifier(x[:, i, :])

            return predict
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def la_transformer(num_classes=1000, loss='softmax', **kwargs):
    # Load pre-trained ViT
    vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    test_only = True
    model = LATransformer(vit_base, lmbd=8, num_classes=num_classes, loss=loss, test_only=test_only)
    return model, InitPretrainedWeights()
