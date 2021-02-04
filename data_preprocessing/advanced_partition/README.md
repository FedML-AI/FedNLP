# Advanced Partition Method
We provide two non-iid partition methods. You can dowoload the our datasets and the partitioned data 
```sh
wget --no-check-certificate --no-proxy https://fednlp.s3-us-west-1.amazonaws.com/data_and_partition_files.zip
unzip data_loaders_and_partition.zip
```
or follow the instruction below to partition your own dataset.

## BERT-based Clustering 
We first use sentence transformer to compute the embedding of the data and then run kmeans to gets K clusters based on client numbers.
### usage 
```sh
python -m data_preprocessing.advanced_partition.kmeans_ex  \
--client_number 100 \
--data_file 'data/data_files/20news_data.h5' \
--partition_file 'data/partition_files/20news_partition.h5' \
--embedding_file data/embedding_files/20news_embedding.h5  \
--task_type text_classification
```
## LDA
we first kmeans to classify data in to 5 clusters and then apply LDA to distribute data in to different number of groups as defined in client number

we already have 5 clusters data for datasets excluding "20news agnews sentiment140 sst2 semeval_2010_task8" because they have their own natural classification. In the each of the rest partition h5 files, you can access the data by the keyword "kmeans_5". If you would like to create different numbers of clusters you can use the kmeans code we provide above
### usage

```sh
python -m data_preprocessing.advanced_partition.lda_ex  \
--client_number 100 \
--data_file 'data/data_files/20news_data.h5' \
--partition_file 'data/partition_files/20news_partition.h5' \
--task_type text_classification \
--min_size 10 \
--alpha 1.0
```
## datasets stats
every data is round to 0.01
### 20news 

|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|18846|18846|
|sample_mean|188.46|188.46|
|sample_std|25.51|66.66|
|std/mean|0.57|0.30|


### agnews
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|127600|127600|
|sample_mean|1276|1276|
|sample_std|520.64|322.76|
|std/mean|0.41|0.25|

### sst2
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|8741|8741|
|sample_mean|87.41|87.41|
|sample_std|49.67|26.39|
|std/mean|0.57|0.30|
### sentiment 140
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|1600498|1600498|
|sample_mean|16004.98|16004.98|
|sample_std|9535.35|5560.68|
|std/mean|0.60|0.35|
### cnn_dailymail
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|312085|312085|
|sample_mean|3120.85|3120.85|
|sample_std|1265.16|888.63|
|std/mean|0.41|0.28|
### cornell_movie_dialogue 
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|221616|221616|
|sample_mean|2216.16|2216.16|
|sample_std|709.90|921.94|
|std/mean|0.32|0.42|
### semeval_2010_task8
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|10717|10717|
|sample_mean|107.17|107.17|
|sample_std|27.87|25.79|
|std/mean|0.26|0.24|
### squad_1.1
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|122325|122325|
|sample_mean|1223.25|1223.25|
|sample_std|415.48|377.13|
|std/mean|0.34|0.31|
### w_nut
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|4681|4681|
|sample_mean|46.81|46.81|
|sample_std|14.08|19.01|
|std/mean|0.30|0.41|
### wikiner
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|||
|sample_mean|||
|sample_std|||
|std/mean|||
### wmt_cs_en
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|218384|218384|
|sample_mean|2183.84|2183.84|
|sample_std|660.36|1269.22|
|std/mean|0.30|0.58|
### wmt_de_en
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|284246|284246|
|sample_mean|2842.46|2842.46|
|sample_std|1194.11|1679.17|
|std/mean|0.42|0.59|
### wmt_ru_en
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|235159|235159|
|sample_mean|2351.59|2351.59|
|sample_std|837.36|1303.71|
|std/mean|0.35|0.55|
### wmt_zh_en
|data|LDA|Kmeans|
|-----| -----| ---|
|client_number|100|100|
|sample_total|252777|252777|
|sample_mean|2527.77|2527.77|
|sample_std|926.80|1274.83|
|std/mean|0.37|0.50|