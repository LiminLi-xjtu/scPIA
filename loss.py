import numpy as np
import torch
import torch.nn as nn
from cal_mmd import *
from utils import euclidean_dist,  compute_optimal_transport
from sklearn.metrics.pairwise import pairwise_distances
from alignment import alignment

class scPIA_Loss(nn.Module):
    def __init__(self):
        super(scPIA_Loss, self).__init__()

    def forward(self, inputs_list, ae_encoded_list, ae_decoded_list, Param):
        lambda1 = Param.lambda1
        lambda2 = Param.lambda2
        mse = nn.MSELoss()

        # Loss 1
        Loss1_1 = mse(inputs_list[0], ae_decoded_list[0])
        Loss1_2 = mse(inputs_list[1], ae_decoded_list[1])
        Loss1 = Loss1_1 + Loss1_2

        # Loss 2
        Loss2 = mmd(ae_encoded_list[0], ae_encoded_list[1])

        # Loss 3
        C = euclidean_dist(ae_encoded_list[0], ae_encoded_list[1])
        P_pred = alignment(C)
        '''
        M = pairwise_distances(ae_encoded_list[0].cpu().detach().numpy(), ae_encoded_list[1].cpu().detach().numpy(), metric='euclidean')
        n, m = M.shape
        r = np.ones(n) #/n
        c = np.ones(m) #/m
        P_pred, _ = compute_optimal_transport(M, r, c, lam=1, epsilon=1e-4)
        P_tmp = P_pred
        '''
        Loss3 = mse(ae_encoded_list[0], torch.mm(P_pred, ae_encoded_list[1]))

        return Loss1 + lambda1 * Loss2 + lambda2 * Loss3, P_pred


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
