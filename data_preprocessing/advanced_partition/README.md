# Advanced Partition Method
We provide two non-iid partition methods. You can dowoload the already partitioned data from amazon s3 link or follow the instruction below to 
partition your own dataset.

## BERT-based Clustering 
We first use sentence transformer to compute the embedding of the data and then run kmeans to gets K clusters based on client numbers.
### usage 

## LDA
we first kmeans to classify data in to 5 clusters and then apply LDA to distribute data in to different number of groups as defined in client number
### usage

