## [Attentive Walk-Aggregating Graph Neural Networks](https://openreview.net/forum?id=TWSTyYd2Rl)  
### Authors: [Mehmet F. Demirel](cs.wisc.edu/~demirel/), [Shengchao Liu](https://chao1224.github.io/), [Siddhant Garg](https://sid7954.github.io/), [Zhenmei Shi](cs.wisc.edu/~zhmeishi/), [Yingyu Liang](cs.wisc.edu/~yliang/)

```
@article{
demirel2022attentive,
title={Attentive Walk-Aggregating Graph Neural Networks},
author={Mehmet F Demirel and Shengchao Liu and Siddhant Garg and Zhenmei Shi and Yingyu Liang},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=TWSTyYd2Rl},
note={}
}
```
--- ---

### Abstract
Graph neural networks (GNNs) have been shown to possess strong representation power, which can be exploited for downstream prediction tasks on graph-structured data, such as molecules and social networks. They typically learn representations by aggregating information from the K-hop neighborhood of individual vertices or from the enumerated walks in the graph. Prior studies have demonstrated the effectiveness of incorporating weighting schemes into GNNs; however, this has been primarily limited to K-hop neighborhood GNNs so far. In this paper, we aim to design an algorithm incorporating weighting schemes into walk-aggregating GNNs and analyze their effect. We propose a novel GNN model, called AWARE, that aggregates information about the walks in the graph using attention schemes. This leads to an end-to-end supervised learning method for graph-level prediction tasks in the standard setting where the input is the adjacency and vertex information of a graph, and the output is a predicted label for the graph. We then perform theoretical, empirical, and interpretability analyses of AWARE. Our theoretical analysis in a simplified setting identifies successful conditions for provable guarantees, demonstrating how the graph information is encoded in the representation, and how the weighting schemes in AWARE affect the representation and learning performance. Our experiments demonstrate the strong performance of AWARE in graph-level prediction tasks in the standard setting in the domains of molecular property prediction and social networks. Lastly, our interpretation study illustrates that AWARE can successfully capture the important substructures of the input graph.

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
