from __future__ import print_function

import argparse
import json
import time
import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from util import *


class AWARE(nn.Module):
    def __init__(self, embed_dim, r_prime, max_walk_len, num_layers, feature_num, out_dim):
        super(AWARE, self).__init__()
        self.r_prime = r_prime
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_walk_len = max_walk_len
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        ##
        self.W = torch.nn.Parameter(torch.randn(embed_dim, feature_num), requires_grad=True)
        self.Wv = torch.nn.Parameter(torch.randn(r_prime, embed_dim), requires_grad=True)
        self.Ww = torch.nn.Parameter(torch.randn(r_prime, r_prime), requires_grad=True)
        self.Wg = torch.nn.Parameter(torch.randn(r_prime, r_prime), requires_grad=True)
        ##
        self.target_model = []
        for i in range(self.num_layers + 1):
            if i == 0:
                self.target_model.append(nn.Linear(r_prime * max_walk_len, r_prime * max_walk_len))
                self.target_model.append(self.relu)
            elif i != num_layers:
                self.target_model.append(
                    nn.Linear(
                        self.target_model[2 * i - 2].out_features,
                        self.target_model[2 * i - 2].out_features // (2 ** (1 - i % 2))
                    )
                )
                self.target_model.append(self.relu)
            else:
                self.target_model.append(nn.Linear(self.target_model[2 * i - 2].out_features, out_dim))
        self.target_model = nn.Sequential(*self.target_model)
        print('MODEL:\n', self.target_model)
        print()
        return

    def forward(self, node_attribute_matrix, adjacent_matrix):
        node_attribute_matrix = torch.matmul(self.W, torch.transpose(node_attribute_matrix, 1, 2))
        F_1 = self.activation(torch.matmul(self.Wv, node_attribute_matrix))

        F_n = F_1
        f_1 = torch.sum(self.activation(torch.matmul(self.Wg, F_n)), dim=2)
        f_T = [f_1]

        for n in range(self.max_walk_len - 1):
            S = torch.bmm(torch.transpose(F_n, 1, 2), torch.matmul(self.Ww, F_n))
            masked_S = S.masked_fill(adjacent_matrix == 0, -1e8)
            A_S = self.softmax(masked_S)
            F_n = torch.bmm(F_n, A_S) * F_1
            f_n = torch.sum(self.activation(torch.matmul(self.Wg, F_n)), dim=2)
            f_T.append(f_n)
        f_T = F.normalize(torch.cat(f_T, dim=1))
        return self.target_model(f_T)


def train():
    best_valid = 1e7
    no_improvement = 0
    for epoch in range(1, 1 + args.epochs):
        model.train()
        train_loss = []
        for batch_id, (graph_id, node_attribute_matrix, adjacent_matrix, label_vector) in enumerate(
                train_dataloader):
            node_attribute_matrix = Variable(node_attribute_matrix).float().to(device)
            adjacent_matrix = Variable(adjacent_matrix).float().to(device)
            trues = label_vector.view(-1, 1).float().to(device) if not multi_class else label_vector.long().to(
                device)

            optimizer.zero_grad()
            preds = model(node_attribute_matrix, adjacent_matrix)
            loss = criterion(preds, trues)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        valid_loss = test(valid_dataloader)

        if valid_loss >= best_valid:
            no_improvement = no_improvement + 1
        else:
            best_valid = valid_loss
            no_improvement = 0

        print('Epoch: {0:d}\tTraining loss: {1:.6f}\tValidation loss: {2:.6f}'
              .format(epoch, train_loss, valid_loss))
        print()
        if epoch % args.every_epoch == 0:
            predict(valid_dataloader, 'valid')
            predict(test_dataloader, 'test')
        if no_improvement > 50:
            print()
            print('***No improvement on the validation loss for 50 epochs, training will be stopped...***')
            break


def test(dataloader):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for batch_id, (graph_id, node_attribute_matrix, adjacent_matrix, label_vector) in enumerate(dataloader):
            node_attribute_matrix = node_attribute_matrix.float().to(device)
            adjacent_matrix = adjacent_matrix.float().to(device)
            trues = label_vector.view(-1, 1).float().to(device) if not multi_class else label_vector.long().to(
                device)
            preds = model(node_attribute_matrix, adjacent_matrix)

            loss = criterion(preds, trues)
            test_loss.append(loss.item())

    test_loss = np.mean(test_loss)
    return test_loss


def predict(dataloader, prefix):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for batch_id, (graph_id, node_attribute_matrix, adjacent_matrix, label_vector) in enumerate(dataloader):
            node_attribute_matrix = node_attribute_matrix.float().to(device)
            adjacent_matrix = adjacent_matrix.float().to(device)
            trues = label_vector.view(-1, 1).float().to(device) if not multi_class else label_vector.long().to(device)
            preds = model(node_attribute_matrix, adjacent_matrix)
            y_true.append(trues)
            y_pred.append(preds)

        y_true = torch.cat(y_true, dim=0).to(device)
        y_pred = torch.cat(y_pred, dim=0).to(device)
        if multi_class:
            _, y_pred = torch.max(y_pred, dim=1)

        if regression:
            return output_regression_result(dataset, y_true, y_pred, prefix)
        else:
            return output_classification_result(dataset, y_true, y_pred, prefix)


if __name__ == '__main__':
    ### Arguments to the program ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='delaney')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--every_epoch', type=int, default=1)
    args = parser.parse_args()
    print('-' * 30)
    print('TASK: ', args.task)

    ### Reprodubility measures ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.index)
    os.environ['PYTHONHASHargs.seed'] = str(args.index)
    np.random.seed(args.index)
    torch.manual_seed(args.index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.index)
        torch.cuda.manual_seed_all(args.index)
    torch.backends.cudnn.deterministic = True
    gpu = False
    if device == torch.device('cuda'):
        gpu = True

    ### Reading hyperparameters ###
    conf = hyper_dict[args.task]
    lr, max_walk_len, r_prime, embed_dim, num_layers = conf['lr'], conf['max_walk_len'], conf['r_prime'], conf[
        'embed_dim'], conf['num_layers']
    print()
    print('HYPERPARAMETERS: {}'.format({'learning rate': lr, 'maximum walk length': max_walk_len, 'r_prime': r_prime,
                                        'embedding dimension': embed_dim, 'epochs': args.epochs,
                                        'number of FC layers': num_layers,
                                        'activation': 'Sigmoid',
                                        'optimizer': 'Adam'}))
    print()

    ### Model and training set up ###
    for ds in dataset_list:
        if args.task in task_dict[ds]:
            dataset = ds
    feature_num = 32 if dataset in ['qm8', 'qm9'] else (
        1 if dataset in ['IMDB-BINARY', 'REDDIT-BINARY', 'IMDB-MULTI', 'COLLAB', 'Mutagenicity'] else 42)
    multi_class = args.task in ['IMDB-MULTI', 'COLLAB']

    regression = False
    if dataset in ['delaney', 'malaria', 'cep', 'qm7', 'qm8', 'qm9']:
        regression = True
        criterion = nn.L1Loss() if dataset in ['qm7', 'qm8', 'qm9'] else nn.MSELoss()
    else:
        criterion = custom_cross_ent(args.task)

    model = AWARE(
        embed_dim=embed_dim,
        r_prime=r_prime,
        max_walk_len=max_walk_len,
        num_layers=num_layers,
        feature_num=feature_num,
        out_dim=3 if args.task in ['IMDB-MULTI', 'COLLAB'] else 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ### Dataloader ###
    print('LOADING DATA...')
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(dataset, args.task, args.index,
                                                                             128)
    print('DATA LOADING DONE!')
    print()

    ### Start Training ###
    print('START TRAINING...')
    print('-' * 30)
    start_time = time.time()
    train()
    end_time = time.time()
    print('-' * 30)
    print('TRAINING DONE!')
    print()
    print('TRAINING DURATION: {} SECONDS'.format(end_time - start_time))
    print()
