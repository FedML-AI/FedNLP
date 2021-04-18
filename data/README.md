# Fednlp datasets

## Introduction

Fednlp datasets contains 12 datasets which can be trained for four popular tasks including:

- **text classification**
- **span extraction**
- **sequence tagging**  
- **sequence-to-sequence**

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
