import torch
from torch import nn
import numpy as np

input1 = np.array([[3,0,0,0],
                   [0,3,3,0],
                   [0,0,0,0],
                   [0,0,0,0]])
input2 = np.array([1,1])

input1 = np.pad(input1,pad_width=1,mode="constant",constant_values=0)
input2 = np.pad(input2,((0,2)),mode="constant",constant_values=0)
input2 = np.reshape(input2,(2,2))
input2 = np.pad(input2,((0,4),(0,4)),mode="constant",constant_values=0)
input = input1+input2
