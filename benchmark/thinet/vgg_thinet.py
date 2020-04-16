import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from compute_flops import *


__all__ = ['thinet30', 'thinet50', 'thinet70', 'resnet50_official']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}
