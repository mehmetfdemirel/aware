from __future__ import print_function
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score

dataset_list = ['delaney', 'malaria', 'cep', 'qm7', 'qm8', 'qm9', 'tox21', 'muv', 'clintox', 'hiv', 'mutagenicity',
                'IMDB-BINARY', 'REDDIT-BINARY', 'IMDB-MULTI', 'COLLAB', 'Mutagenicity']
task_dict = {
    'tox21':
        [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ],
    'muv':
        [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
        ],
    'hiv': ['hiv'],
    'clintox': ['CT_TOX', 'FDA_APPROVED'],
    'mutagenicity': ['mutagenicity'],
    'IMDB-BINARY': ['IMDB-BINARY'],
    'REDDIT-BINARY': ['REDDIT-BINARY'],
    'IMDB-MULTI': ['IMDB-MULTI'],
    'COLLAB': ['COLLAB'],
    'Mutagenicity': ['Mutagenicity'],
    'delaney': ['delaney'],
    'malaria': ['malaria'],
    'cep': ['cep'],
    'qm7': ['qm7'],
    'qm8':
        [
            'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0',
            'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'
        ],
    'qm9':
        [
            'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0',
            'u298', 'h298', 'g298'
        ],
}

hyper_dict = {
	"delaney": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 500, 'embed_dim': 100, 'num_layers': 1},
	"malaria": {'lr': 0.001, 'max_walk_len': 12, 'r_prime': 300, 'embed_dim': 100, 'num_layers': 1},
	"cep": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 100, 'embed_dim': 100, 'num_layers': 1},
	"qm7": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 300, 'num_layers': 2},
	"E1-CC2": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 300, 'embed_dim': 100, 'num_layers': 3},
	"E2-CC2": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 500, 'embed_dim': 500, 'num_layers': 3},
	"f1-CC2": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 100, 'embed_dim': 500, 'num_layers': 3},
	"f2-CC2": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 300, 'embed_dim': 500, 'num_layers': 3},
	"E1-PBE0": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 500, 'embed_dim': 300, 'num_layers': 2},
	"E2-PBE0": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 500, 'embed_dim': 100, 'num_layers': 3},
	"f1-PBE0": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 100, 'num_layers': 3},
	"f2-PBE0": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 500, 'num_layers': 3},
	"E1-CAM": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 300, 'embed_dim': 100, 'num_layers': 3},
	"E2-CAM": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 300, 'embed_dim': 300, 'num_layers': 2},
	"f1-CAM": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 500, 'num_layers': 3},
	"f2-CAM": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 300, 'embed_dim': 300, 'num_layers': 3},
	"NR-AR": {'lr': 0.001, 'max_walk_len': 12, 'r_prime': 100, 'embed_dim': 100, 'num_layers': 2},
	"NR-AR-LBD": {'lr': 0.001, 'max_walk_len': 9, 'r_prime': 100, 'embed_dim': 100, 'num_layers': 3},
	"NR-AhR": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 100, 'embed_dim': 100, 'num_layers': 1},
	"NR-Aromatase": {'lr': 0.0001, 'max_walk_len': 9, 'r_prime': 500, 'embed_dim': 500, 'num_layers': 1},
	"NR-ER": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 500, 'embed_dim': 500, 'num_layers': 1},
	"NR-ER-LBD": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 500, 'num_layers': 1},
	"NR-PPAR-gamma": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 500, 'embed_dim': 300, 'num_layers': 2},
	"SR-ARE": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 500, 'embed_dim': 300, 'num_layers': 1},
	"SR-ATAD5": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 100, 'embed_dim': 100, 'num_layers': 2},
	"SR-HSE": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 500, 'embed_dim': 100, 'num_layers': 1},
	"SR-MMP": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 500, 'embed_dim': 300, 'num_layers': 1},
	"SR-p53": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 100, 'num_layers': 1},
	"MUV-466": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 300, 'embed_dim': 300, 'num_layers': 3},
	"MUV-548": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 300, 'num_layers': 1},
	"MUV-600": {'lr': 0.0001, 'max_walk_len': 9, 'r_prime': 300, 'embed_dim': 500, 'num_layers': 2},
	"MUV-644": {'lr': 0.001, 'max_walk_len': 9, 'r_prime': 100, 'embed_dim': 300, 'num_layers': 3},
	"MUV-652": {'lr': 0.0001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 500, 'num_layers': 1},
	"MUV-689": {'lr': 0.0001, 'max_walk_len': 6, 'r_prime': 300, 'embed_dim': 300, 'num_layers': 1},
	"MUV-692": {'lr': 0.001, 'max_walk_len': 9, 'r_prime': 500, 'embed_dim': 100, 'num_layers': 2},
	"MUV-712": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 100, 'embed_dim': 100, 'num_layers': 1},
	"MUV-713": {'lr': 0.0001, 'max_walk_len': 12, 'r_prime': 500, 'embed_dim': 500, 'num_layers': 2},
	"MUV-733": {'lr': 0.0001, 'max_walk_len': 9, 'r_prime': 500, 'embed_dim': 300, 'num_layers': 2},
	"MUV-737": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 100, 'embed_dim': 500, 'num_layers': 3},
	"MUV-810": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 100, 'embed_dim': 500, 'num_layers': 3},
	"MUV-832": {'lr': 0.001, 'max_walk_len': 12, 'r_prime': 300, 'embed_dim': 100, 'num_layers': 1},
	"MUV-846": {'lr': 0.001, 'max_walk_len': 9, 'r_prime': 300, 'embed_dim': 300, 'num_layers': 3},
	"MUV-852": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 500, 'embed_dim': 500, 'num_layers': 2},
	"MUV-858": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 500, 'embed_dim': 300, 'num_layers': 2},
	"MUV-859": {'lr': 0.0001, 'max_walk_len': 6, 'r_prime': 500, 'embed_dim': 500, 'num_layers': 2},
	"CT_TOX": {'lr': 0.001, 'max_walk_len': 9, 'r_prime': 500, 'embed_dim': 300, 'num_layers': 1},
	"FDA_APPROVED": {'lr': 0.001, 'max_walk_len': 9, 'r_prime': 500, 'embed_dim': 500, 'num_layers': 2},
	"hiv": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 500, 'embed_dim': 500, 'num_layers': 1},
	"Mutagenicity": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 100, 'num_layers': 1},
	"IMDB-BINARY": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 100, 'embed_dim': 100, 'num_layers': 1},
	"IMDB-MULTI": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 300, 'embed_dim': 100, 'num_layers': 1},
	"REDDIT-BINARY": {'lr': 0.001, 'max_walk_len': 3, 'r_prime': 100, 'embed_dim': 100, 'num_layers': 1},
	"COLLAB": {'lr': 0.001, 'max_walk_len': 6, 'r_prime': 500, 'embed_dim': 100, 'num_layers': 1},
}



class MoleculeDataset(Dataset):
    def __init__(self, node_attribute_matrix_list, adjacent_matrix_list, label_list):
        self.node_attribute_matrix_list = node_attribute_matrix_list
        self.adjacent_matrix_list = adjacent_matrix_list
        self.label_list = label_list

    def __len__(self):
        return len(self.node_attribute_matrix_list)

    def __getitem__(self, idx):
        node_attribute_matrix = torch.from_numpy(self.node_attribute_matrix_list[idx])
        adjacent_matrix = torch.from_numpy(self.adjacent_matrix_list[idx])
        label = self.label_list[idx]
        return node_attribute_matrix, adjacent_matrix, label


class GeneralDataset(Dataset):
    def __init__(self, graph_id_list, node_attribute_matrix_list, adjacent_matrix_list, label_list):
        self.graph_id_list = graph_id_list
        self.node_attribute_matrix_list = node_attribute_matrix_list
        self.adjacent_matrix_list = adjacent_matrix_list
        self.label_list = label_list

    def __len__(self):
        return len(self.node_attribute_matrix_list)

    def __getitem__(self, idx):
        if self.graph_id_list is None:
            graph_id = -1
        else:
            graph_id = self.graph_id_list[idx]
        node_attribute_matrix = torch.from_numpy(self.node_attribute_matrix_list[idx])
        adjacent_matrix = torch.from_numpy(self.adjacent_matrix_list[idx])
        label = self.label_list[idx]
        return graph_id, node_attribute_matrix, adjacent_matrix, label


def create_dataloaders(dataset, task, running_index, batch_size):
    data_loaders = []
    for data_type in ['train', 'valid', 'test']:
        folder = dataset + '/' + task if dataset in ['muv', 'tox21', 'clintox'] else dataset
        data_path = 'datasets/{}/{}/{}_graph.npz'.format(folder, running_index, data_type)
        graph_id_list, adjacent_matrix_list, node_attribute_matrix_list, label_list = get_data(data_path, dataset, task)
        temp_dataset = GeneralDataset(
            graph_id_list=graph_id_list if dataset in ['IMDB-BINARY', 'REDDIT-BINARY', 'IMDB-MULTI', 'COLLAB',
                                                       'Mutagenicity'] else None,
            node_attribute_matrix_list=node_attribute_matrix_list,
            adjacent_matrix_list=adjacent_matrix_list,
            label_list=label_list)
        data_loaders.append(torch.utils.data.DataLoader(temp_dataset, batch_size=batch_size, shuffle=True))
    return data_loaders


def get_data(data_path, dataset, task):
    data = np.load(data_path, allow_pickle=True)
    graph_id_list = None
    if dataset in ['IMDB-BINARY', 'REDDIT-BINARY', 'IMDB-MULTI', 'COLLAB', 'Mutagenicity']:
        graph_id_list = data['graph_id_list']
    adjacent_matrix_list = data['adjacent_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']

    if dataset in ['qm8', 'qm9']:
        label_list = data[task]
    elif dataset in ['IMDB-BINARY', 'REDDIT-BINARY', 'IMDB-MULTI', 'COLLAB', 'Mutagenicity']:
        label_list = data['label_list']
    else:
        label_list = data['label_name']

    return graph_id_list, adjacent_matrix_list, node_attribute_matrix_list, label_list


def custom_cross_ent(task):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if task not in ['IMDB-BINARY', 'REDDIT-BINARY', 'IMDB-MULTI', 'COLLAB', 'Mutagenicity']:
        with open("datasets/balance.json", 'r') as f:
            weights = json.load(f)

        neg_to_pos_ratio = torch.FloatTensor([1 - weights[task]]).to(device)
        loss_criterion = nn.BCEWithLogitsLoss(pos_weight=neg_to_pos_ratio)
    else:
        if task in ['IMDB-MULTI', 'COLLAB']:
            loss_criterion = nn.CrossEntropyLoss()
        else:
            loss_criterion = nn.BCEWithLogitsLoss()
    return loss_criterion


def output_classification_result(dataset, trues, preds, datatype):
    if preds is not None:
        preds = preds.detach().cpu()
        trues = trues.detach().cpu()
        roc_val = roc_auc_single(preds, trues)
        acc = accuracy(preds, trues, 3 if dataset in ['IMDB-MULTI', 'COLLAB'] else 2)
        if dataset not in ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB', 'Mutagenicity']:
            print('{} roc-auc: {}'.format(datatype, roc_val))
        print('{} accuracy: {}'.format(datatype, acc))
        return roc_val
    return None


def output_regression_result(dataset, trues, preds, datatype):
    def output(y_true, y_pred, mode):
        rmse = rmse_score(y_true, y_pred)
        mae = mae_score(y_true, y_pred)
        print('{} rmse: {}'.format(mode, rmse))
        print('{} mae: {}'.format(mode, mae))
        if dataset in ['qm7', 'qm8', 'qm9']:
            return mae
        else:
            return rmse

    if preds is not None:
        preds = preds.detach().cpu()
        trues = trues.detach().cpu()
        return output(trues, preds, datatype)


def rmse_score(y_true, y_pred):
    '''Computes RMSE error.'''
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_score(y_true, y_pred):
    '''Computes MAE.'''
    return mean_absolute_error(y_true, y_pred)


def roc_auc_single(predicted, actual):
    try:
        auc_ret = roc_auc_score(actual, predicted)
    except ValueError:
        auc_ret = np.nan

    return auc_ret


def accuracy(predicted, actual, num_classes):
    try:
        if num_classes == 2:
            predicted = nn.Sigmoid()(predicted)
            predicted_labels = torch.clamp(torch.floor(num_classes * predicted), min=0.0, max=num_classes - 1).float()
        else:
            predicted_labels = predicted
        acc = accuracy_score(actual, predicted_labels)
    except ValueError:
        acc = np.nan
    return acc

