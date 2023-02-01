import random

import numpy as np
from tqdm import tqdm
from visualization import visualize
import torch

from model import AE
from loss import AverageMeter, scPIA_Loss
from datasets import load_data


def main(Params):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # seed
    random.seed(Params.seed)
    np.random.seed(Params.seed)
    torch.manual_seed(Params.seed)
    torch.random.manual_seed(Params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Params.seed)
        torch.backends.cudnn.deterministic = True

    # load dataset
    print("load data ...")
    X_list, Y_list = load_data(Params)

    print(Params.data_name + ', view size:', Params.view_size) # print('training samples', train_X_list[0].shape[0]

    # network
    print("build model ...")
    # network architecture
    arch_list = []
    for view in range(Params.view_size):
        arch = [X_list[view].shape[1]]
        arch.extend(Params.arch)
        arch_list.append(arch)

    model = AE(arch_list).to(device)
    criterion = scPIA_Loss().to(device)

    # data tensor
    train_X_list = []
    train_X_list.append(torch.from_numpy(X_list[0]).to(device))
    train_X_list.append(torch.from_numpy(X_list[1]).to(device))

    optimizer_train = torch.optim.Adam(model.parameters(), lr=Params.lr, weight_decay=1e-5)
    Z, P = train(Params, model, optimizer_train, train_X_list, criterion, device)


    result = dict(
        {"data": X_list, "label": Y_list, 'fea': Z, 'pred': P})



    return X_list, Y_list, Z, P




def train(Params, model, optimizer, X_list, criterion, device):
    print('training ...')

    model.train()
    losses = AverageMeter()
    t_progress = tqdm(range(Params.epoch), desc='Training')
    Z = []
    P = []
    for epoch in t_progress:
        current_loss = 0
        count = 0
        ae_encoded, ae_decoded = model(X_list)
        loss, P_pred = criterion(X_list, ae_encoded, ae_decoded, Params)


        losses.update(loss.item())
        current_loss += loss.item()
        count += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch % 100 == 0):
            Z.append(ae_encoded)
            P.append(P_pred.cpu().detach().numpy())

        # loging
        t_progress.write('epoch %d : loss %.6f' % (epoch, current_loss / count))
        t_progress.set_description_str(' Loss=' + str(loss.item()))

    return Z, P
