import numpy as np
import pandas as pd
import cv2
import glob
import functools
import tqdm
import PIL
import h5py
from multiprocess import Pool
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import time
import torch
import scipy
import scipy.misc
import skimage
import skimage.transform as skt
import math

from torchvision import transforms

# import torch.functional as F
import torch.nn.functional as F

eps = np.finfo(np.float64).eps

plt.rcParams['figure.figsize'] = 10, 10


'''
Affine crop testing
'''

img_t = F.interpolate(transforms.ToTensor()(scipy.misc.face()).unsqueeze(0), size=(768, 768), mode='bilinear')

theta = torch.from_numpy(np.array([[[0.2, 0.0, 0.2], [0.0, 0.2, 0.3]]]))

grid = F.affine_grid(theta, torch.Size((1, 3, 768, 768)))
grid.size()

plt.rcParams['figure.figsize'] = 8, 8

fig, axis = plt.subplots(nrows=1, ncols=2)

axis[0].imshow(grid[0, :, :, 0])
axis[0].set_title('x')

axis[1].imshow(grid[0, :, :, 1])
axis[1].set_title('y')
plt.show()

G = torch.bmm(grid[:, :, 0, 1].unsqueeze(2), grid[:, 0, :, 0].unsqueeze(1))
G.size()
plt.imshow(G.squeeze())

'''
Grid created by Jason for cropping
'''

# the grid is created by an inner product across a linspace in x & y
N = 1
W, H = 128, 128
scale = 0.7

theta2 = torch.from_numpy(np.array([[0.0, 0.5, 0.2]])).type(torch.float32)

base_grid = [torch.linspace(-1, 1, W).expand(N, W),
             torch.linspace(-1, 1, H).expand(N, H)]
m_x = torch.max(torch.zeros_like(base_grid[0]),
                1 - torch.abs(theta2[:, 1].unsqueeze(1) - base_grid[0])).unsqueeze(2)
m_y = torch.max(torch.zeros_like(base_grid[1]),
                1 - torch.abs(theta2[:, 2].unsqueeze(1) - base_grid[1])).unsqueeze(1)

# the final mask is a batch-matrix-multiple of each inner product
# the dimensions expected here are m_x = [N, W, 1] & m_y = [N, 1, H]
M = torch.bmm(m_x, m_y)

M_clip = torch.clamp(M, M.max() * scale, M.max())

plt.rcParams['figure.figsize'] = 8, 8

fig, axis = plt.subplots(nrows=1, ncols=2)

axis[0].imshow(M.squeeze())
axis[0].set_title('m')

axis[1].imshow(M_clip.squeeze())
axis[1].set_title('clip')
plt.show()

'''
crop with affine grid sampler.
'''

# Affine upper left and right bottom coords
x1, y1 = 50, 80
x2, y2 = 100, 120

# find bb

"""
[  x2-x1             x1 + x2 - W + 1  ]
[  -----      0      ---------------  ]
[  W - 1                  W - 1       ]
[                                     ]
[           y2-y1    y1 + y2 - H + 1  ]
[    0      -----    ---------------  ]
[           H - 1         H - 1      ]
"""
height, width = img_t.size(2), img_t.size(3)

# requires grad
theta = torch.zeros(1, 2, 3)

# do crop
theta[:, 0, 0] = (x2 - x1) / (width - 1)
theta[:, 0, 2] = (x1 + x2 - width + 1) / (width - 1)
theta[:, 1, 1] = (y2 - y1) / (height - 1)
theta[:, 1, 2] = (y1 + y2 - height + 1) / (height - 1)

grid = F.affine_grid(theta, torch.Size((1, 3, 768, 768)))
crop = F.grid_sample(img_t, grid.type(torch.float32), padding_mode='zeros')

plt.rcParams['figure.figsize'] = 8, 8
with sns.axes_style('white'):
    fig, axis = plt.subplots(nrows=1, ncols=2)

    axis[0].imshow(img_t.data.numpy().squeeze(0).transpose((1, 2, 0)))
    axis[0].plot(x1, y1, 'r+')
    axis[0].plot(x2, y2, 'b+')
    axis[0].set_title('')

    axis[1].imshow(crop.data.numpy().squeeze(0).transpose((1, 2, 0)))
    axis[1].set_title('')


'''
unit test grid sampler versus affine crop
'''

img_t = F.interpolate(transforms.ToTensor()(scipy.misc.face()).unsqueeze(0), size=(768, 768), mode='bilinear')
im = img_t.squeeze().transpose(0, 2).transpose(0, 1).numpy()


def grid_cropper(img_t, theta, h=768, w=768):
    grid = F.affine_grid(theta, torch.Size((1, 3, h, w)))
    crop = F.grid_sample(img_t, grid.type(torch.float32), padding_mode='zeros')
    return crop


def param2theta(param, w, h):
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + param[0, 0] + param[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + param[1, 0] + param[1, 1] - 1
    return theta


def theta2param(theta, w, h):
    param = np.zeros((3, 3))
    param[-1, :] = [0, 0, 1]
    param[0, 0] = theta[0, 0]
    param[0, 1] = theta[0, 1] * w / h
    param[0, 2] = theta[0, 2] * w / 2 - param[0, 0] - param[0, 1] + 1
    param[1, 0] = theta[1, 0] * h / w
    param[1, 1] = theta[1, 1]
    param[1, 2] = theta[1, 2] * h / 2 - param[1, 0] - param[1, 1] + 1
    return param


im_org = img_t.squeeze().transpose(0, 2).transpose(0, 1).numpy()
h, w = im_org.shape[0], im_org.shape[1]

# TFORM -> GRID
tform = skt.SimilarityTransform(scale=0.2, rotation=0.0, translation=(10, 10))

img1 = skt.warp(im_org, tform, output_shape=(w, h))

theta = param2theta(tform.params, w, h)

theta = torch.from_numpy(theta).unsqueeze(0).type(torch.float)

print(tform.params)
print(theta)

img2 = grid_cropper(img_t, theta)
img2 = img2.data.cpu().numpy().squeeze(0).transpose((1, 2, 0))


# GRID -> TFORM
# theta = torch.from_numpy(np.array([[[0.2, 0.0, 0.2], [0.0, 0.2, 0.3]]]))

img3 = grid_cropper(img_t, theta)
img3 = img3.data.numpy().squeeze(0).transpose((1, 2, 0))

param = theta2param(theta.data.numpy().squeeze(), w, h)

print(tform.params)
print(theta)

img4 = skt.warp(im_org, np.linalg.inv(param), output_shape=(w, h))


plt.rcParams['figure.figsize'] = 8, 8
with sns.axes_style('white'):
    fig, axis = plt.subplots(nrows=2, ncols=2)

    axis[0, 0].imshow(img1)
    axis[0, 0].set_title('affine')

    axis[0, 1].imshow(img2)
    axis[0, 1].set_title('grid')

    axis[1, 0].imshow(img4)
    axis[1, 0].set_title('affine')

    axis[1, 1].imshow(img3)
    axis[1, 1].set_title('grid')
