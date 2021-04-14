import h5py
import argparse
import numpy as np
import json
from sklearn.model_selection import train_test_split
import math
from decimal import * 

getcontext().prec=128

def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    '''
    params
    ------------------------------------
    N : int  total length of the dataset
    alpha : int  similarity of each client, the larger the alpha the similar data for each client
    client_num : int number of clients
    idx_batch: 2d list shape(client_num, ?), this is the list of index for each client
    idx_k : 1d list  list of index of the dataset
    ------------------------------------

    return
    ------------------------------------
    idx_batch : 2d list shape(client_num, ?) list of index for each client
    min_size : minimum size of all the clients' sample 
    ------------------------------------
    '''
    # first shuffle the index 
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))
    # get the index in idx_k according to the dirichlet distribution 
    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the new batch list for each client
    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

    return idx_batch


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--client_number', type=int, default='100', metavar='CN',
                        help='client number for lda partition')

    parser.add_argument('--data_file', type=str, default='data/data_files/20news_data.h5',
                        metavar="DF", help='data pickle file path')

    parser.add_argument('--partition_file', type=str, default='data/partition_files/20news_partition.h5',
                        metavar="PF", help='partition pickle file path')
    
    parser.add_argument('--task_type', type=str, metavar="TT", help='task type')

    parser.add_argument('--min_size', type=int, metavar="MS", help='minimal size of each client sample')

    parser.add_argument('--kmeans_num', type=int, metavar="KN", help='number of k-means cluster')

    parser.add_argument('--alpha', type=float, metavar="A", help='alpha value for LDA')
    
    args = parser.parse_args()


    print("start reading data")
    
    
    client_num = args.client_number
    alpha = args.alpha # need adjustment for each dataset

    min_size = 0
    max_size = 0
    batch_size = []
    label_vocab = []
    label_assignment = np.array([])

    print('retrive data')
    # retrive total index length 
    data = h5py.File(args.data_file,"r")
    attributes = json.loads(data["attributes"][()])
    total_index_len = len(attributes['index_list'])

    if args.task_type == 'text_classification':
            label_vocab = list(set([data['Y'][i][()] for i in data['Y'].keys()]))
            label_assignment = np.array([data['Y'][i][()] for i in data['Y'].keys()])
    else:
            partition = h5py.File(args.partition_file,"r")
            label_vocab = list(set(partition["kmeans_%d"%args.kmeans_num+'/client_assignment'][()]))
            label_assignment = np.array(partition["kmeans_%d"%args.kmeans_num+'/client_assignment'][()])
            partition.close()

    data.close()

    partition_pkl = [[] for _ in range(client_num)]
    label_client_matrix = [[] for _ in label_vocab]
    label_proportion = []
    print('start process')

    # shuffle indexs and calculate each label proportion of the dataset
    for index, value in enumerate(label_vocab):
        label_location = np.where(label_assignment == value)[0]
        label_proportion.append(len(label_location) / len(label_assignment))
        np.random.shuffle(label_location)
        label_client_matrix[index].extend(label_location)
        np.random.shuffle(label_location)
        label_client_matrix[index].extend(label_location)
        

    index_marker = np.zeros(len(label_vocab),dtype=int)
    total_index = total_index_len
    each_client_length = int(total_index / client_num)
    client_length = 0
    print(total_index)
    
    lable_dis = []
    for i in partition_pkl:
        client_dir_dis = np.array([alpha for i in label_vocab])
        proportions = np.random.dirichlet(client_dir_dis) # still has to be number of classes
        client_length = min(each_client_length,total_index)
        if total_index < client_length * 2:
            client_length = total_index
        total_index -= client_length
        temp_client_length = client_length
        temp = {}

        for index, value in enumerate(label_vocab):
            offset = round(proportions[index] * client_length)
            if offset >= temp_client_length:
                offset = temp_client_length
                temp_client_length = 0
            else:
                if index == (len(label_vocab) - 1):
                    offset = temp_client_length
                temp_client_length -= offset
            
            temp[index] = offset
            start = int(index_marker[index])
            end = int(index_marker[index] + offset)
            i.extend(label_client_matrix[index][start:end])
            index_marker[index] = index_marker[index] + offset
        lable_dis.append(temp)


    print("start dirichlet distribution")
    print([len(i) for i in label_client_matrix])
    print(index_marker)
    for i in lable_dis:
        print(i.values())
        
    exit()
    
    # add 
    print("store data in h5 data")
    partition = h5py.File(args.partition_file,"a")

    if('/lda' in partition ):
        del partition['/lda']

    if('/niid_%f'%args.alpha in partition):
        del partition['/niid_%f'%args.alpha]

    if('/niid_%.1f'%args.alpha in partition):
        del partition['/niid_%.1f'%args.alpha]
    
    partition[ '/niid_%.1f'%args.alpha + '/n_clients'] = client_num
    partition['/niid_%.1f'%args.alpha + '/alpha'] = alpha
    for i, data in enumerate(partition_pkl):
        train, test = train_test_split(partition_pkl[i], test_size=0.2, train_size = 0.8, random_state=42)
        train_path = '/niid_%.1f'%args.alpha + '/partition_data/'+str(i)+'/train/'
        test_path = '/niid_%.1f'%args.alpha + '/partition_data/'+str(i)+'/test/'
        partition[train_path] = train
        partition[test_path] = test
    partition.close()

main()


