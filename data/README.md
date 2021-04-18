# Fednlp datasets

## Introduction

Fednlp datasets contains 12 datasets which can be trained for four popular tasks including:

- **text classification**
- **span extraction**
- **sequence tagging**  
- **sequence-to-sequence**

## RawDataLoader
RawDataLoader is responsible for loading data from original data files as well as saving the data as a special format(h5py in our case). As we know, every dataset has its own data format which means RawDataLoader should be various depending on the dataset. However, it is hard to load data from dataset-specific h5py files since different datasets have different structure in their h5py files and the framework has to spend a lot of time for maintaining a large bunch of dataset-specific loading functions. Therefore, we decide to create a more general way to keep our framework as clean as possible. 

RawDataLoader is splitted into four classes based on the task definition in FedNLP. Each class has task-specific attributes and functions for loading and saving data. For each dataset, users can check their task definition, inherit one of the RawDataLoader classes and fulfill the required attributes and functions. Following the above structure, the framework can exploit the data in a faster and more convenient way.

As we mentioned before, FedNLP utilizes h5py as a specific format to store data for future usage. There are several reasons why FedNLP uses h5py and why FedNLP does not load the data from original data files directly. The first one is about indexing. In FedNLP, we have to simulate the setting in federated learning. That is, partitioning the dataset into a number of parts as clients and training the machine learning model in those clients separately. In this case, each client only needs to load a part of data in a dataset, which is h5py good at. h5py is able to load a part of data instead of taking the whole dataset into memory by employing indexing techniques. In addition, h5py files can be used by multi processes which is suitable for our based framework FedML. Because a process means a client or a server in FedML so that all clients are capable of getting touch with data simultaneously.



## Download Data

### Bash downloader
We provide each dataset with a bash script to download the raw data within its own directory.


### Pre-processed h5py files 

You can also download all h5py files by using following commands, which is also in `download_data.sh` and `download_partiton.sh`.
```bash
DATA_DIR=~/fednlp_data/	# Change to your local data folder.
rm -rf ${DATA_DIR}/data_files
rm -rf ${DATA_DIR}/partition_files
declare -a data_names=("20news" "agnews" "cnn_dailymail" "cornell_movie_dialogue" "semeval_2010_task8" "sentiment140" "squad_1.1" "ploner" "sst_2" "wikiner" "wmt_cs-en" "wmt_de-en" "wmt_ru-en" "wmt_zh-en")

mkdir ${DATA_DIR}/data_files
mkdir ${DATA_DIR}/partition_files
for data_name in "${data_names[@]}"
do
	wget --no-check-certificate --no-proxy -P ${DATA_DIR}/data_files https://fednlp.s3-us-west-1.amazonaws.com/data_files/${data_name}_data.h5
	wget --no-check-certificate --no-proxy -P ${DATA_DIR}/partition_files https://fednlp.s3-us-west-1.amazonaws.com/partition_files/${data_name}_partition.h5
done
```
We provide two kinds of h5py files: the one in direcotry **data_files**  contains the whole dataset.
The other files in directory **partition_files** are splited dataset using multiple partition methods.
