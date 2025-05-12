# Ghiotto Andrea   2118418

import os, cv2, glob, math, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from IPython import display

from sklearn.neighbors import KDTree
import random

from torch.utils.tensorboard import SummaryWriter