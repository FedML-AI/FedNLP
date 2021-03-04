import h5py
import os
import pickle
from sentence_transformers import SentenceTransformer   
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np



class partition_method:
    def __init__(self, client_num, data_path, partition_path, partition_method, task_type, \
                embedding_path=None, batch_size=None, min_size=None, alpha_value=None):
        self.client_num = client_num
        self.data_path = data_path
        self.partition_path = partition_path
        self.partition_method = partition_method
        self.task_type = task_type
        self.batch_size = batch_size
        self.min_size = min_size
        self.alpha_value = alpha_value
        self.embedding_path = embedding_path
        if embedding_path is not None and os.path.exists(self.embedding_path) == False:
            with open(self.embedding_path,'wb') as f:
                pass
        if  os.path.exists(self.partition_path) == False:
            f = h5py.File(self.partition_path, "w")
            f.close()

    def partition(self):
        if self.partition_method == "kmeans":
            self.kmeans()
        elif self.partition_method == 'lda':
            self.lda()
        elif self.partition_method == 'uniform':
            self.uniform()

    def kmeans(self):
        pass

    def lda(self):
        pass
    def uniform(self):
        pass