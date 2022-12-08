import datasets as datasets
import torch
from torchvision import datasets

train = torch.utils.data.DataLoader(
    datasets.MNIST('.', download=True))
