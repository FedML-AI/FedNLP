import os
import data_preprocessing.news_20.data_loader
from data_preprocessing.base.utils import *
from model.fed_transformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


dataset_name = "20news"
data_file = "data/data_loaders/20news_data_loader.pkl"
partition_file = "data/partition/20news_partition.pkl"
partition_method = "uniform"
batch_size = 32

data_loader = None
print("load_data. dataset_name = %s" % dataset_name)
data_loader = data_preprocessing.news_20.data_loader.ClientDataLoader(
    data_file, partition_file, partition_method=partition_method, tokenize=False)

train_data = data_loader.get_train_batch_data()
test_data = data_loader.get_test_batch_data()
data_attr = data_loader.get_attributes()
labels_map = data_attr["target_vocab"]
num_labels = len(labels_map)


train_data = [(x, labels_map[y]) for x, y in zip(train_data["X"], train_data["Y"])]
train_df = pd.DataFrame(train_data)

test_data = [(x, labels_map[y]) for x, y in zip(test_data["X"], test_data["Y"])]
test_df = pd.DataFrame(test_data)

# Create a ClassificationModel
model = ClassificationModel(
    "distilbert", "distilbert-base-uncased", num_labels=num_labels,
    args={"num_train_epochs": 3, "learning_rate": 1e-5, "do_lower_case": True, "manual_seed": 42,
          "reprocess_input_data": True, "overwrite_output_dir": True, "fp16": True, "wandb_project": "fednlp"})

model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

print(result)

# pip install --upgrade wandb
# wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
#


# print(dataset[:3])


# python -m experiments.centralized.fed_transformer_exps.text_classification
