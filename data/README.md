# Fednlp datasets

## Intro

Fednlp datasets contains 12 datasets which can be trained for four popular tasks including:

- **text classification**
- **span extraction**
- **sequence tagging**  
- **sequence-to-sequence**

## Download Data

### Bash downloader
We provide each dataset with a bash script to download the raw data within its own directory.


### Pre-processed Pickles 

You can also download all h5py files by using following commands.
```
rm -rf data_files
rm -rf partition_files
declare -a data_names=("20news" "agnews" "cnn_dailymail" "cornell_movie_dialogue" 
	"semeval_2010_task8" "sentiment140" "squad_1.1" "sst_2" "w_nut" "wikiner" 
	"wmt_cs-en" "wmt_de-en" "wmt_ru-en" "wmt_zh-en")

mkdir data_files
mkdir partition_files
for data_name in "${data_names[@]}"
do
	wget --no-check-certificate --no-proxy -P ./data_files https://fednlp.s3-us-west-1.amazonaws.com/data_files/${data_name}_data.h5
	wget --no-check-certificate --no-proxy -P ./partition_files https://fednlp.s3-us-west-1.amazonaws.com/partition_files/${data_name}_partition.h5
done
```
We provide two kinds of h5py files: the one in direcotry **data_loaders**  contains the whole dataset.
The other files in directory **partition** are splited dataset using multiple partition methods.
You need to put the `data_loaders` and `partition` here.
