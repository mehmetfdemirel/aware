# [An Analysis of Attentive Walk-Aggregating Graph Neural Networks](https://arxiv.org/pdf/2110.02667.pdf)  
### Authors: [Mehmet F. Demirel](cs.wisc.edu/~demirel/), [Shengchao Liu](https://chao1224.github.io/), [Siddhant Garg](https://sid7954.github.io/), [Yingyu Liang](cs.wisc.edu/~yliang/)
--- ---

### Abstract
Graph neural networks (GNNs) have been shown to possess strong representation power, which can be exploited for downstream prediction tasks on graph-structured data, such as molecules and social networks. They typically learn representations by aggregating information from the K-hop neighborhood of individual vertices or from the enumerated walks in the graph. Prior studies have demonstrated the effectiveness of incorporating weighting schemes into GNNs; however, this has
been primarily limited to K-hop neighborhood GNNs so far. In this paper, we aim to extensively analyze the effect of incorporating weighting schemes into walkaggregating GNNs. Towards this objective, we propose a novel GNN model, called AWARE, that aggregates information about the walks in the graph using attention
schemes in a principled way to obtain an end-to-end supervised learning method for graph-level prediction tasks. We perform theoretical, empirical, and interpretability analyses of AWARE. Our theoretical analysis provides the first provable guarantees for weighted GNNs, demonstrating how the graph information is encoded in the representation, and how the weighting schemes in AWARE affect the representation and learning performance. We empirically demonstrate the superiority of AWARE over prior baselines in the domains of molecular property prediction (61 tasks) and social networks (4 tasks). Our interpretation study illustrates that AWARE can successfully learn to capture the important substructures of the input graph.

---

### Instructions to run the code

#### 1. Create and activate Anaconda environment
```
cd env
bash create_env.sh
```

#### 2. Download datasets and process them for experiments
```
cd datasets
bash download_data.sh
bash run_preprocess.sh
```
#### 3. Run AWARE
* Specify arguments
  ```
  export task=<TASK_NAME>
  export index=<RUNNING_INDEX>
  export output=<OUTPUT_FOLDER>
  ```

* Run the code via
  ```
  mkdir -p "$output"/"$task"
  python train.py --task="$task" --index="$index" > "$output"/"$task"/"$index".out
  ```

  You can also do the following without specifying any arguments.
    ```
    bash run.sh
    ```
