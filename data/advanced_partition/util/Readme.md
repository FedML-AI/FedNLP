## Usage


partition_name:{kemeans, lda}
task_name is shown in the title of the figure so give any name is fine

visualization_heterogeneity cannot be applied on kmeans because we use Kmeans cluster as labels in LDA
therefore we assume all the data in the same client in kmeans_partition has the same label.




``` bash
DATA_DIR=~/fednlp_data/

python -m data_preprocessing.advanced_partition.util.visualization_heterogeneity \
--partition_name lda \
--partition_file '${DATA_DIR}/partition_files/20news_partition.h5' \
--data_file '${DATA_DIR}/data_files/20news_data.h5' \
--task_name 20news \
--figure_path 'data_preprocessing/advanced_partition/heatmap_figure' \
--task_type text_classification 

python -m data_preprocessing.advanced_partition.util.visualization_heterogeneity \
--partition_name lda \
--partition_file '${DATA_DIR}/partition_files/wikiner_partition.h5' \
--data_file '${DATA_DIR}/data_files/wikiner_data.h5' \
--task_name wikiner \
--figure_path 'data_preprocessing/advanced_partition/heatmap_figure' \
--task_type name_entity_recognition

python -m data_preprocessing.advanced_partition.util.visualization_stats \
--partition_name kmeans \
--partition_file '${DATA_DIR}/partition_files/wikiner_partition.h5' \
--task_name wikiner \
--figure_path 'data_preprocessing/advanced_partition/heatmap_figure'

python -m data_preprocessing.advanced_partition.util.visualization_heatmap \
--partition_name lda \
--client_num 100 \ 
--partition_file '${DATA_DIR}/partition_files/wikiner_partition.h5' \
--data_file '${DATA_DIR}/data_files/wikiner_data.h5' \
--task_name wikiner \
--figure_path 'data_preprocessing/advanced_partition/heatmap_figure' \
--task_type name_entity_recognition
```