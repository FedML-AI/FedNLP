## Usage

partition_name:{kemeans, lda}
task_name is shown in the title of the figure so give any name is fine

visualization_heterogeneity cannot be applied on kmeans because we use Kmeans cluster as labels in LDA
therefore we assume all the data in the same client in kmeans_partition has the same label.

``` bash
python -m data_preprocessing.advanced_partition.util.visualization_heterogeneity \
--partition_name lda \
--partition_file data/partition_files/wikiner_partition.h5 \
--task_name wikiner
```


```bash
python -m data_preprocessing.advanced_partition.util.visualization_stats \
--partition_name kmeans \
--partition_file data/partition_files/wikiner_partition.h5 \
--task_name wikiner

```