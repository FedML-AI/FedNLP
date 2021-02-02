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
wget --no-check-certificate --no-proxy https://fednlp.s3-us-west-1.amazonaws.com/data_and_partition_files.zip
unzip data_and_partition_files.zip
```
We provide two kinds of h5py files: the one in direcotry **data_loaders**  contains the whole dataset.
The other files in directory **partition** are splited dataset using multiple partition methods.
You need to put the `data_loaders` and `partition` here.
