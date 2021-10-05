#!/usr/bin/env bash

mkdir -p logs

molecular_datasets=(delaney malaria cep qm7 qm8 qm9 tox21 clintox muv hiv)
TU_datasets=(IMDB-BINARY IMDB-MULTI REDDIT-BINARY COLLAB Mutagenicity)

for dataset in "${molecular_datasets[@]}"; do
  python process_"$dataset".py > logs/"$dataset".processed
done

for dataset in "${TU_datasets[@]}"; do
  python process_TUDataset.py --dataset_name="$dataset" > logs/"$dataset".processed
done
