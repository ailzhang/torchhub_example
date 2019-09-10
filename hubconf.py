dependencies = ['torch']
import torch
from model import Net

def mnist(pretrained=False):
    m = Net()
    if pretrained:
        m.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/ailzhang/torchhub_example/releases/download/0.1/mnist_init_ones'))
    return m
