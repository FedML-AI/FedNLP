import h5py
import argparse
from sentence_transformers import SentenceTransformer   
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def get_embedding_Kmeans(corpus, N_clients, bsz=16):
    embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device = 'cuda:0') # server only

    corpus_embeddings = embedder.encode(corpus, show_progress_bar=True, batch_size=bsz) #smaller batch size for gpu

    embedding_data = {}
    embedding_data['data'] = corpus_embeddings

    ### KMEANS clustering
    num_clusters = N_clients
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    return cluster_assignment, embedding_data



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--client_number', type=int, default='100', metavar='CN',
                        help='client number for lda partition')
    parser.add_argument('--bsz', type=int, default='16', metavar='CN',
                        help='batch size for sentenceBERT')

    parser.add_argument('--data_file', type=str, default='data/data_files/wikiner_data.h5',
                        metavar="DF", help='data pickle file path')

    parser.add_argument('--partition_file', type=str, default='data/partition_files/wikiner_partition.h5',
                        metavar="PF", help='partition pickle file path')

    parser.add_argument('--embedding_file', type=str, default='data/embedding_files/wikiner_embedding.h5',
                        metavar="EF", help='embedding pickle file path')
    
    parser.add_argument('--task_type', type=str, metavar="TT", default="text_classfication", help='task type')
    # add a stroe_true for --overwrite 
    
    args = parser.parse_args()

    N_Clients = args.client_number
    # print(111)
    # exit()

    f = h5py.File(args.data_file,"r")
    corpus = []
    if args.task_type == 'name_entity_recognition': # specifically wnut and wikiner datesets
        for i in f['X'].keys():
            sentence = f['X'][i][()]
            sentence = [i.decode('UTF-8') for i in sentence]
            corpus.append(" ".join(sentence))

    elif args.task_type == 'reading_comprehension': # specifically Squad1.1 dataset
        for i in f['context_X'].keys():
            sentence = f['context_X'][i][()]
            corpus.append(sentence)
            
    else:
        for i in f['X'].keys():
            sentence = f['X'][i][()]
            print(sentence)
            corpus.append(sentence)
    f.close()

    cluster_assignment, embedding_data = get_embedding_Kmeans(corpus, N_Clients, args.bsz)

    # TODO: add a file existence checker
    f = h5py.File(args.embedding_file,"w")
    f['/embedding_data'] = embedding_data
    f.close()

    partition_pkl = {}
    for index, idx in enumerate(cluster_assignment):
        if idx in partition_pkl :
            partition_pkl[idx].append(index)
        else:
            partition_pkl[idx] = [index]

    print(partition_pkl[0])

    partition = ""
    partition = h5py.File(args.partition_file,"w")

    partition['/kmeans/n_clients'] = args.client_number
    for i in sorted(partition_pkl.keys()):
        train, test = train_test_split(partition_pkl[i], test_size=0.4, train_size = 0.6, random_state=42)
        train_path = '/kmeans/partition_data/'+str(i)+'/train/'
        test_path = '/kmeans/partition_data/'+str(i)+'/test/'
        partition[train_path] = train
        partition[test_path] = test
    partition.close()


main()