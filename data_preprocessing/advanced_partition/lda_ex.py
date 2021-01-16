import pickle
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

    parser.add_argument('--data_file', type=str, default='data/data_loaders/20news_data_loader.pkl',
                        metavar="DF", help='data pickle file path')

    parser.add_argument('--partition_file', type=str, default='data/partition/20news_partition.pkl',
                        metavar="PF", help='partition pickle file path')
    
    parser.add_argument('--task_type', type=str, metavar="TT", help='task type')

    parser.add_argument('--min_size', type=int, metavar="MS", help='minimal size of each client sample')

    parser.add_argument('--alpha', type=float, metavar="A", help='alpha value for LDA')
    
    args = parser.parse_args()

    data = ""
    with open(args.data_file,'rb') as f:
        data = pickle.load(f)

    client_num = args.client_number
    labels = list(set(data['Y']))
    label_list = np.array(data['Y'])

    N = len(data['Y'])
    alpha = args.alpha # need adjustment for each dataset

    partition_pkl = [[] for _ in range(client_num)]
    min_size = 0

    if args.task_type == 'classification':
        for i in labels:
            idx_k = np.where(label_list == i)[0]
            while min_size < args.min_size:
                partition_pkl = [[] for _ in range(client_num)]
                partition_pkl, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                                partition_pkl, idx_k)
    else:
        # aasume all data have the same label so no need to update seperately 
        idx_k = np.array(data['attributes']['index_list'])
        while min_size < args.min_size:
                partition_pkl = [[] for _ in range(client_num)]
                partition_pkl, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                                partition_pkl, idx_k)

    partition = ""
    with open(args.partition_file,'rb') as f:     
        partition = pickle.load(f)

    partition['lda'] = {}
    partition['lda']['n_clients'] = client_num
    partition['lda']['partition_data'] = {}
    # add 

    for i, data in enumerate(partition_pkl):
        partition['lda']['partition_data'][i] = {}
        train, test = train_test_split(data, test_size=0.4, train_size = 0.6, random_state=42)
        partition['lda']['partition_data'][i]['train'] = train
        partition['lda']['partition_data'][i]['test'] = test

    print(partition['kmeans']['partition_data'])
    with open(args.partition_file,'wb') as f:
        pickle.dump(partition, f, pickle.HIGHEST_PROTOCOL) 

if __name__ == "__main__":
    main()


