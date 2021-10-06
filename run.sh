#!/usr/bin/env bash

task_list=(CT_TOX FDA_APPROVED NR-AR NR-AR-LBD
NR-AhR NR-Aromatase NR-ER NR-ER-LBD NR-PPAR-gamma
SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53 MUV-466
MUV-548 MUV-600 MUV-644 MUV-652 MUV-689 MUV-692
MUV-712 MUV-713 MUV-733 MUV-737 MUV-810 MUV-832
MUV-846 MUV-852 MUV-858 MUV-859 hiv
delaney malaria cep qm7 E1-CC2 E2-CC2 f1-CC2
f2-CC2 E1-PBE0 E2-PBE0 f1-PBE0 f2-PBE0 E1-CAM
E2-CAM f1-CAM f2-CAM mu alpha homo lumo gap r2
zpve cv u0 u298 h298 g298
IMDB-BINARY IMDB-MULTI REDDIT-BINARY COLLAB Mutagenicity)

output="OUTPUT"
index_list=(0 1 2 3 4)

for task in "${task_list[@]}"; do
  for index in "${index_list[@]}"; do
          mkdir -p "$output"/"$task"
          python train.py \
            --task="$task" \
            --index="$index" > "$output"/"$task"/"$index".out
  done
done
