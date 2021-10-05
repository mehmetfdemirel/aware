from __future__ import print_function

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from preprocess import *
import os
from sklearn.model_selection import train_test_split

num_of_splits = 5
max_atom_num = 55
qm9_tasks = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298',
             'u298_atom', 'h298_atom', 'g298_atom']


def prepare(dataset_name, split, clean_mols=False):
    whole_data_pd = pd.read_csv('{}.csv'.format(dataset_name))

    column = ['smiles'] + qm9_tasks
    data_pd = whole_data_pd.dropna(how='any', subset=column)[column]
    data_pd.columns = ['SMILES'] + qm9_tasks
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
    data_pd = data_pd[['SMILES', 'Fingerprints'] + qm9_tasks]

    if not os.path.exists('{}/{}'.format(dataset_name, split)):
        os.makedirs('{}/{}'.format(dataset_name, split))
    print('total shape\t', data_pd.shape)

    suppl = Chem.SDMolSupplier('{}.sdf'.format(dataset_name), clean_mols, False, False)
    molecule_list = [mol for mol in suppl]

    raw_df = pd.read_csv('{}.sdf.csv'.format(dataset_name))
    print(raw_df.shape, '\t', raw_df.columns)

    indices = np.arange(data_pd.shape[0])
    train_indices, temp_testvalid = train_test_split(indices, train_size=0.8)
    test_indices, valid_indices = train_test_split(temp_testvalid, train_size=0.5)

    inds = [train_indices, test_indices, valid_indices]

    for index in range(len(inds)):
        partition = data_pd.iloc[indices[inds[index]]]
        print('{}_set'.format(names[index]), '\t', partition.shape)

        partition.to_csv('{}/{}/{}.csv.gz'.format(dataset_name, split, names[index]), compression='gzip', index=None)

        w = Chem.SDWriter('{}/{}/{}.sdf'.format(dataset_name, split, names[index]))
        for t in inds[index]:
            w.write(molecule_list[t])
            w.flush()

        set_pd = raw_df.iloc[inds[index]]
        set_pd.to_csv('{}/{}/{}.sdf.csv'.format(dataset_name, split, names[index]), index=None)

    if split == num_of_splits - 1:
        partition = data_pd.iloc[indices]
        print('full {} dataset'.format(dataset_name), '\t', partition.shape)

        partition.to_csv('{}/full_{}.csv.gz'.format(dataset_name, dataset_name), compression='gzip', index=None)

        w = Chem.SDWriter('{}/full_{}.sdf'.format(dataset_name, dataset_name))
        for t in indices:
            w.write(molecule_list[t])
            w.flush()

        set_pd = raw_df.iloc[indices]
        set_pd.to_csv('{}/full_{}.sdf.csv'.format(dataset_name, dataset_name), index=None)

    return


if __name__ == '__main__':
    dataset_name = 'qm9'
    names = ["train", "test", "valid"]

    for split in range(num_of_splits):
        print()
        print("split {}".format(split))
        print("--------------------------------")
        np.random.seed((split + 1) * 123)
        prepare(dataset_name, split)

        for i in range(len(names)):
            extract_graph_multi_tasks_SDF(data_path='{}/{}/{}.csv.gz'.format(dataset_name, split, names[i]),
                                          sdf_data_path='{}/{}/{}.sdf'.format(dataset_name, split, names[i]),
                                          out_file_path='{}/{}/{}_graph.npz'.format(dataset_name, split, names[i]),
                                          task_list=qm9_tasks,
                                          max_atom_num=max_atom_num)

        print("--------------------------------")
    
    extract_graph_multi_tasks_SDF(data_path='{}/full_{}.csv.gz'.format(dataset_name, dataset_name),
                                  sdf_data_path='{}/full_{}.sdf'.format(dataset_name, dataset_name),
                                  out_file_path='{}/full_graph.npz'.format(dataset_name),
                                  task_list=qm9_tasks,
                                  max_atom_num=max_atom_num)


