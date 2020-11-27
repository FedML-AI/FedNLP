# Introduction
This is FedNLP, an application ecosystem for natural language processing based the FedML framework (https://github.com/FedML-AI/FedML).

FedNLP uses FedML repository as the git submodule. In other words, FedNLP only focus on adavanced models and dataset, while FedML supports various
federated optimizers (e.g., FedAvg) and platforms (Distributed Computing, IoT/Mobile, Standalone).

# Installation
http://doc.fedml.ai/#/installation

After the clone of this repository, please run the following command to get `FedML` submodule to your local.
```
git submodule init
git submodule update
```

# Code Structure of FedNLP
Note: please make sure that the code in FedNLP only uses `FedML/fedml_core` and `FedML/fedml_api`.
In near future, once FedML is stable, we will release it as a python package. 
At that time, we can install FedML package with pip or conda without the need to use Git submodule.

`FedML`: a soft repository link generated using `git submodule add https://github.com/FedML-AI/FedML`.


`data`: provide data downloading scripts and store the downloaded datasets.
Note that in `FedML/data`, there also exists datasets for research, but these datasets are used for evaluating federated optimizers (e.g., FedAvg) and platforms.
FedNLP supports more advanced datasets and models.

`data_preprocessing`: data loaders

`model`: advanced NLP models.

`trainer`: please define your own `trainer.py` by inheriting the base class in `FedML/fedml-core/trainer/fedavg_trainer.py`.
Some tasks can share the same trainer.

`experiments/distributed`: 
1. `experiments` is the entry point for training. It contains experiments in different platforms. We start from `distributed`.
1. Every experiment integrates FOUR building blocks `FedML` (federated optimizers), `data_preprocessing`, `model`, `trainer`.
2. To develop new experiments, please refer the code at `experiments/distributed/text-classification`.

`experiments/centralized`: 
1. please provide centralized training script in this directory. 
2. This is used to get the reference model accuracy for FL. 
3. You may need to accelerate your training through distributed training on multi-GPUs and multi-machines. Please refer the code at `experiments/centralized/DDP_demo`.

# Update FedML Submodule
If you need new features supported by FedML framework, please update FedML submodule using Git commands at this tutorial:
https://git-scm.com/book/en/v2/Git-Tools-Submodules

Method 1: 
```
git fetch
git merge origin/master
```

Method 2:
```
cd FedML
git submodule update --remote FedML
```



# Citation
Please cite our FedNLP and FedML paper if it helps your research.
You can describe us in your paper like this: "We develop our experiments based on FedNLP [1] and FedML [2]".

 