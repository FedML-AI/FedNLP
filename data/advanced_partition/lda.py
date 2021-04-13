import h5py
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

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
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


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

    partition_pkl = [[] for _ in range(client_num)]
    min_size = 0

    

    print("start dirichlet distribution")
    while min_size < args.min_size:
        partition_pkl = [[] for _ in range(client_num)]
        if args.task_type == 'text_classification':
            data = h5py.File(args.data_file,"r")
            N = len(data['Y'])
            labels = list(set([data['Y'][i][()] for i in data['Y'].keys()]))
            label_list = np.array([data['Y'][i][()] for i in data['Y'].keys()])
            data.close()
            for i in labels:
                idx_k = np.where(label_list == i)[0]
                partition_pkl, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                                    partition_pkl, idx_k)
        else:
            # aasume all data have the same label so no need to update seperately 
            data = h5py.File(args.data_file,"r")
            N = len(data['Y'])
            data.close()
            partition = h5py.File(args.partition_file,"r")
            labels = list(set(partition["kmeans_%d"%args.kmeans_num+'/client_assignment'][()]))
            label_list = np.array(partition["kmeans_%d"%args.kmeans_num+'/client_assignment'][()])
            for i in labels:
                idx_k = np.where(label_list == i)[0]
                partition_pkl, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                                        partition_pkl, idx_k)
            partition.close()
    # add 
    print("store data in h5 data")
    partition = h5py.File(args.partition_file,"a")
    if('/lda' in partition):
        del partition['/lda']

    partition['lda/n_clients'] = client_num
    partition['lda/alpha'] = alpha
    for i, data in enumerate(partition_pkl):
        train, test = train_test_split(partition_pkl[i], test_size=0.2, train_size = 0.8, random_state=42)
        train_path = '/lda/partition_data/'+str(i)+'/train/'
        test_path = '/lda/partition_data/'+str(i)+'/test/'
        partition[train_path] = train
        partition[test_path] = test
    partition.close()

main()


