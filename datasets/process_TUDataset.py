import os
import os.path as osp
import shutil
import numpy as np
import argparse

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split


class TUDataset(InMemoryDataset):
    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False,
                 processed_filename='data.pt'):
        self.name = name
        self.processed_filename = processed_filename
        super(TUDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            self.data.x = self.data.x[:, self.num_node_attributes:]

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            if self.data.x[:, i:].sum().item() == self.data.x.size(0):
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return self.processed_filename

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self[0].num_node_features

    def download(self):
        temp = '{}/{}.zip'.format(self.url, self.name)
        print('temp\t', temp)
        path = download_url('{}/{}.zip'.format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


class Graph:
    def __init__(self, data, max_node_num):
        x, edge_index, y = data.x, data.edge_index, data.y
        self.node_num = x.shape[0]
        self.feature_dim = x.shape[1]

        self.node_attribute_matrix = np.zeros((max_node_num, self.feature_dim))
        self.node_attribute_matrix[0:self.node_num, :] = x.numpy()
        # for i in range(self.node_num):
        #     assert np.sum(self.node_attribute_matrix[i, :]) > 0
        for i in range(self.node_num, max_node_num):
            assert np.sum(self.node_attribute_matrix[i, :]) == 0

        self.adjacency_matrix = np.zeros((max_node_num, max_node_num))
        for u, v in zip(edge_index[0], edge_index[1]):
            self.adjacency_matrix[u][v], self.adjacency_matrix[v][u] = 1, 1

        return


class Degree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.x = deg.view(-1, 1)
        return data


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def transform(dataset, dataset_name):
    '''
    transform from pytorch-geometric dataset into our own format.
    '''
    ########## get degree as features ##########
    max_degree = 0
    degs = []
    invalid_graph_id_list = []
    for idx, data in enumerate(dataset):
        temp = degree(data.edge_index[0], dtype=torch.long)
        # If the degree is too large, or if the number of node is too large, remove this graph.
        if dataset_name in ['REDDIT-BINARY', 'COLLAB']:
            threshold = 200 if dataset_name == 'REDDIT-BINARY' else 100
            if (temp.max().item() > threshold) or (data.edge_index[0].max().item() > threshold):
                invalid_graph_id_list.append(idx)
                print('Invalid {}, {}'.format(idx, data))
                continue
        degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, degs[-1].max().item())
    # TODO: need to fine-tune the featurization
    deg = torch.cat(degs, dim=0).to(torch.float)
    mean, std = deg.mean().item(), deg.std().item()
    dataset.transform = NormalizedDegree(mean, std)
    # dataset.transform = Degree(mean, std)
    print('max_degree\t', max_degree)
    print('invalid graph\t', invalid_graph_id_list)

    ########## get some statistics ##########
    max_num_nodes = num_nodes = num_edges = 0
    for idx, data in enumerate(dataset):
        if idx in invalid_graph_id_list:
            continue
        num_nodes += data.num_nodes
        num_edges += data.num_edges
        max_num_nodes = max(max_num_nodes, data.num_nodes)
    print('Name', dataset)
    print('Graphs', len(dataset))
    print('Nodes', num_nodes / len(dataset))
    print('Max nodes', max_num_nodes)
    print('Edges', (num_edges // 2) / len(dataset))
    print('Features', dataset.num_features)
    print('Classes', dataset.num_classes)
    print()

    ########## start padding ##########
    N = len(dataset)
    adjacent_matrix_list, node_attribute_matrix_list, label_list = [], [], []
    for i in range(N):
        if i in invalid_graph_id_list:
            continue
        data = dataset[i]
        graph = Graph(data=data, max_node_num=max_num_nodes)
        adjacent_matrix_list.append(graph.adjacency_matrix)
        node_attribute_matrix_list.append(graph.node_attribute_matrix)
        label_list.append(data.y.item())

    adjacent_matrix_list = np.stack(adjacent_matrix_list, axis=0)
    node_attribute_matrix_list = np.stack(node_attribute_matrix_list, axis=0)
    label_list = np.stack(label_list, axis=0)
    print(np.max(label_list))

    print('adjacent_matrix_list: ', adjacent_matrix_list.shape)
    print('node_attribute_matrix_list: ', node_attribute_matrix_list.shape)
    print('label_list', label_list.shape)

    ########## start padding ##########
    num_of_splits = 5
    for split in range(num_of_splits):
        if not os.path.exists('{}/{}'.format(dataset_name, split)):
            os.makedirs('{}/{}'.format(dataset_name, split))
        np.random.seed((split + 1) * 123)

        train_data, temp_data = train_test_split(np.arange(adjacent_matrix_list.shape[0]), train_size=0.8,
                                                 stratify=label_list)
        test_data, valid_data = train_test_split(temp_data, train_size=0.5, stratify=label_list[temp_data])

        set_names = ['train', 'valid', 'test']
        sets = [train_data, valid_data, test_data]
        print('len of train: {}\tlen of val: {}\tlen of test: {}.'.format(len(train_data), len(valid_data),
                                                                          len(test_data)))

        for i in range(3):
            np.savez_compressed(
                '{}/{}/{}_graph.npz'.format(dataset_name, split, set_names[i]),
                graph_id_list=sets[i],
                adjacent_matrix_list=adjacent_matrix_list[sets[i]],
                node_attribute_matrix_list=node_attribute_matrix_list[sets[i]],
                label_list=label_list[sets[i]],
                indices=sets[i])

    np.savez_compressed(
        '{}/full_graph.npz'.format(dataset_name),
        graph_id_list=sets[i],
        adjacent_matrix_list=adjacent_matrix_list,
        node_attribute_matrix_list=node_attribute_matrix_list,
        label_list=label_list)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='IMDB-BINARY')
    args = parser.parse_args()
    dataset_name = args.dataset_name

    print('Constructing {} ====='.format(dataset_name))
    if dataset_name in ['IMDB-BINARY', 'REDDIT-BINARY']:
        dataset = TUDataset(root=dataset_name, name=dataset_name)
        transform(dataset, dataset_name)

    print('Done with dataset {}.\n\n\n'.format(dataset_name))

