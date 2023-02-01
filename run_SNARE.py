import os, sys

from main import main
import numpy as np
from visualization import visualize
# CUDA env
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Parameter settings
class Params():
    seed = 123
    data_name = 'SNARE'
    view_size = 2

    lambda1 = 1
    lambda2 = 10
    lr = 0.001
    epoch = 2000
    arch = [32, 32, 5]


current_dir = sys.path[0]
Params.main_dir = current_dir

params = Params()
train_X_list, train_Y_list, Z, P = main(params)

result = dict(
    {"data": train_X_list, "label": train_Y_list, 'fea': Z, 'pred': P})

datatype = [result['label'][0], result['label'][1]]
for i in range(len(result['fea'])):
    features1 = result['fea'][i][0].cpu().detach().numpy()
    features2 = result['fea'][i][1].cpu().detach().numpy()
    P_pred = result['pred'][i]
    inte = []
    inte.append(features1)
    inte.append(np.dot(P_pred, features2))
    visualize([result['data'][0], result['data'][1]], inte, datatype, mode='TSNE')

np.save("result_SNARE.npy", result)