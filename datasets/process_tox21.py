from __future__ import print_function

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, MolFromMolBlock, MolToSmarts
from preprocess import *
import os
from sklearn.model_selection import train_test_split

num_of_splits = 5
target_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
max_atom_num = 55


def prepare_fingerprints_TOX21(dataset_name, split):
    whole_data_pd = pd.read_csv('{}.csv.gz'.format(dataset_name))

    for target_name in target_names:
        print(target_name)
        column = [target_name, 'mol_id', 'smiles']
        data_pd = whole_data_pd.dropna(how='any', subset=column)[column]
        data_pd = data_pd.rename(columns={"smiles": "SMILES", "mol_id": "Molecule"})

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

        data_pd = data_pd.loc[valid_index]
        data_pd['Fingerprints'] = morgan_fps
        data_pd = data_pd[['Molecule', 'SMILES', 'Fingerprints', target_name]]

        y_label = data_pd[target_name].tolist()
        y_label = np.array(y_label)

        directory = '{}/{}/{}'.format(dataset_name, target_name, split)
        if not os.path.exists(directory):
            os.makedirs(directory)

        print('total shape\t', data_pd.shape)

        train_set, temp_testvalid = train_test_split(data_pd, train_size=0.8, stratify=y_label)
        y_label = temp_testvalid[target_name]
        test_set, valid_set = train_test_split(temp_testvalid, train_size=0.5, stratify=y_label)

        sets = [train_set, test_set, valid_set]

        for t in range(len(sets)):
            print("{} set.shape = {}".format(names[t], sets[t].shape))
            sets[t].to_csv('{}/{}.csv.gz'.format(directory, names[t]), compression='gzip', index=None)

        if split == num_of_splits - 1:
            data_pd.to_csv('{}/{}/full_{}.csv.gz'.format(dataset_name, target_name, target_name), compression='gzip', index=None)
    return


def get_hit_ratio():
    for target_name in target_names:
        directory = '{}/{}'.format(dataset_name, target_name)
        data_path = '{}/full_graph.npz'.format(directory)
        y_label = []
        data = np.load(data_path)
        y_label.extend(data['label_name'])
        hit_ratio = 1.0 * sum(y_label) / len(y_label)
        print('\'{}\': {},'.format(target_name, hit_ratio))


if __name__ == '__main__':
    dataset_name = 'tox21'
    names = ["train", "test", "valid"]

    for split in range(num_of_splits):
        print()
        print("split {}".format(split))
        print("--------------------------------")
        np.random.seed((split + 1) * 123)
        prepare_fingerprints_TOX21(dataset_name, split)

        for target_name in target_names:
            directory = '{}/{}'.format(dataset_name, target_name)
            for i in range(len(names)):
                extract_graph(data_path='{}/{}/{}.csv.gz'.format(directory, split, names[i]),
                              out_file_path='{}/{}/{}_graph.npz'.format(directory, split, names[i]),
                              label_name=target_name,
                              max_atom_num=max_atom_num)

        print("--------------------------------")

    print()
    print('full {} dataset:'.format(dataset_name))
    for target_name in target_names:
        directory = '{}/{}'.format(dataset_name, target_name)
        for i in range(len(names)):
            extract_graph(data_path='{}/full_{}.csv.gz'.format(directory, target_name),
                          out_file_path='{}/full_graph.npz'.format(directory),
                          label_name=target_name,
                          max_atom_num=max_atom_num)
    get_hit_ratio()
