from __future__ import print_function

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, MolFromMolBlock, MolToSmarts
from preprocess import *
import os
from sklearn.model_selection import train_test_split

num_of_splits = 5
max_atom_num = 55


def prepare_fingerprints_delaney(dataset_name, split):
    target_name = dataset_name
    whole_data_pd = pd.read_csv('./{}-processed.csv'.format(dataset_name))

    column = ['measured log solubility in mols per litre', 'Compound ID', 'smiles']
    data_pd = whole_data_pd.dropna(how='any', subset=column)[column]
    data_pd.columns = [target_name, 'Molecule', 'SMILES']
    print(data_pd.columns)

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
    data_pd = data_pd[['Molecule', 'SMILES', 'Fingerprints', target_name]]

    if not os.path.exists('{}/{}'.format(dataset_name, split)):
        os.makedirs('{}/{}'.format(dataset_name, split))
    print('total shape\t', data_pd.shape)
    
    train_set, temp_testvalid = train_test_split(data_pd, train_size=0.8)
    test_set, valid_set = train_test_split(temp_testvalid, train_size=0.5)

    sets = [train_set, test_set, valid_set]

    for t in range(len(sets)):
        print("{} set.shape = {}".format(names[t], sets[t].shape))
        sets[t].to_csv('{}/{}/{}.csv.gz'.format(dataset_name, split, names[t]), compression='gzip', index=None)

    if split == num_of_splits - 1:
        data_pd.to_csv('{}/full_{}.csv.gz'.format(dataset_name, dataset_name), compression='gzip', index=None)


    return


if __name__ == '__main__':
    dataset_name = 'delaney'
    names = ["train", "test", "valid"]

    for split in range(num_of_splits):
        print()
        print("split {}".format(split))
        print("--------------------------------")
        np.random.seed((split + 1) * 123)
        prepare_fingerprints_delaney(dataset_name, split)

        for i in range(len(names)):
            extract_graph(data_path='{}/{}/{}.csv.gz'.format(dataset_name, split, names[i]),
                          out_file_path='{}/{}/{}_graph.npz'.format(dataset_name, split, names[i]),
                          label_name='delaney',
                          max_atom_num=max_atom_num)

        print("--------------------------------")

    print('full {} dataset: '.format(dataset_name))
    extract_graph(data_path='{}/full_{}.csv.gz'.format(dataset_name, dataset_name),
                  out_file_path='{}/full_graph.npz'.format(dataset_name),
                  label_name='delaney',
                  max_atom_num=max_atom_num)



