from __future__ import print_function

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, MolFromMolBlock, MolToSmarts
from preprocess import *
import os
from sklearn.model_selection import train_test_split

num_of_splits = 5
max_atom_num = 100


def prepare_fingerprints_HIV(dataset_name, split):
    whole_data_pd = pd.read_csv('{}.csv'.format(dataset_name))
    print(whole_data_pd.shape, '\t', whole_data_pd.columns)

    column = ['HIV_active', 'smiles']
    data_pd = whole_data_pd.dropna(how='any', subset=column)[column]
    data_pd = data_pd.rename(columns={'smiles': 'SMILES', 'HIV_active': dataset_name})

    morgan_fps = []
    valid_index = []

    index_list = data_pd.index.tolist()
    smiles_list = data_pd['SMILES'].tolist()
    for idx, smiles in zip(index_list, smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if len(mol.GetAtoms()) > max_atom_num:
            print('Outlier {} has {} atoms'.format(idx, mol.GetNumAtoms()))
            continue
        valid_index.append(idx)
        fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        morgan_fps.append(fingerprints.ToBitString())

    data_pd = data_pd.iloc[valid_index]
    data_pd['Fingerprints'] = morgan_fps
    data_pd = data_pd[['SMILES', 'Fingerprints', dataset_name]]

    y_label = data_pd[dataset_name].tolist()
    y_label = np.array(y_label)

    directory = '{}/{}'.format(dataset_name, split)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('total shape\t', data_pd.shape)

    train_set, temp_testvalid = train_test_split(data_pd, train_size=0.8, stratify=y_label)
    y_label = temp_testvalid[dataset_name]
    test_set, valid_set = train_test_split(temp_testvalid, train_size=0.5, stratify=y_label)

    sets = [train_set, test_set, valid_set]

    for t in range(len(sets)):
        print("{} set.shape = {}".format(names[t], sets[t].shape))
        sets[t].to_csv('{}/{}.csv.gz'.format(directory, names[t]), compression='gzip', index=None)

    if split == num_of_splits - 1:
        data_pd.to_csv('{}/full_{}.csv.gz'.format(dataset_name, dataset_name), compression='gzip', index=None)

    return


def get_hit_ratio():
        data_path = '{}/full_graph.npz'.format(dataset_name)
        y_label = []
        data = np.load(data_path)
        y_label.extend(data['label_name'])
        hit_ratio = 1.0 * sum(y_label) / len(y_label)
        print('\'{}\': {},'.format(dataset_name, hit_ratio))


if __name__ == '__main__':
    dataset_name = 'hiv'
    names = ["train", "test", "valid"]

    for split in range(num_of_splits):
        print()
        print("split {}".format(split))
        print("--------------------------------")
        np.random.seed((split + 1) * 123)
        prepare_fingerprints_HIV(dataset_name, split)

        directory = '{}/{}'.format(dataset_name, split)
        for i in range(len(names)):
            extract_graph(data_path='{}/{}.csv.gz'.format(directory, names[i]),
                          out_file_path='{}/{}_graph.npz'.format(directory, names[i]),
                          label_name=dataset_name,
                          max_atom_num=max_atom_num)

        print("--------------------------------")

    print('full {} dataset: '.format(dataset_name))
    extract_graph(data_path='{}/full_{}.csv.gz'.format(dataset_name, dataset_name),
                  out_file_path='{}/full_graph.npz'.format(dataset_name),
                  label_name=dataset_name,
                  max_atom_num=max_atom_num)

    get_hit_ratio()
