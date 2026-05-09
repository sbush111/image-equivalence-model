'''
Data set reference:
Krizhevsky, A., Nair, V., and Hinton, G. CIFAR-100 python version [dataset]. 
University of Toronto. https://www.cs.toronto.edu/~kriz/cifar.html

The file accessible on the previous link is a picked object, and unpickling it 
would be insecure. As such, we will instead use the CIFAR100 object built
in to torchvision.
'''

from torchvision.datasets import CIFAR100

_ = CIFAR100(root='data', download=True)