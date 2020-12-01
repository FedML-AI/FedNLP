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

You can also download pickle files for each dataset in this [link](https://drive.google.com/folderview?id=1OhZ5NDaVz0VZX5jy8V_I_sfR25R2k_OE).  
We provide two kinds of pickle files: the one in direcotry **data loaders**  contains the whole dataset.
The other files in directory **partition** are splited dataset using multiple partition methods.
You need to put the `data_loaders` and `partition` here.
