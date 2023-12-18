# CIDR: A Cooperative Integrated Dynamic Refining Method for Minimal Feature Removal Problem
Code implementing "CIDR: A Cooperative Integrated Dynamic Refining Method for Minimal Feature
Removal Problem"



## Overview

The minimal feature removal problem in the post-hoc explanation area aims to identify the minimal feature set (MFS). Prior studies using the greedy algorithm to calculate the minimal feature set lack the exploration of feature interactions under a monotonic assumption  which cannot be satisfied in general scenarios. In order to address the above limitations, 
we  propose a Cooperative Integrated Dynamic Refining method (CIDR) to efficiently  discover  minimal feature sets. Specifically, we design Cooperative Integrated Gradients (CIG) to detect interactions between features. By incorporating CIG and the characteristics of minimal feature set, we transform the minimal feature removal problem into a knapsack problem. Additionally, we  devise an auxiliary
Minimal Feature Refinement algorithm to determine the  minimal feature set from numerous candidate sets.
To the best of our knowledge, our work is the first to address the minimal feature removal problem in the field of natural language processing. Extensive experiments demonstrate that CIDR is capable of tracing more semantically representative minimal feature sets with improved interpretability across various models and datasets.



 ## Usage:
- Create the following folder structure.
```
Scripts
    │
    ├── data
    ├── experiments
    
```
- To run experiment cd  experiments 
 ### For BERT/SST2
-  test:  ```python test.py --data_dir path_to_sst2 --task_name sst-2 --bert_path path_to_bert --tokenizer_path path_to_bert --num_classes 2 --model_type bert --model_name_or_path bert-base-uncased --output_dir ./output/sst2/bert --savename bestmodel.pt --cal_AOPC --useDIG --pad_token del --seed 926 --reload --iter_epochs 5 --is_second --use_cls --max_seq_length 32 --evaluate_amount 1821 --k 5 --per_gpu_eval_batch_size 64```


