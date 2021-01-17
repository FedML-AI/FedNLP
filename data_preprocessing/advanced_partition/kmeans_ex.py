import pickle
import argparse
from sentence_transformers import SentenceTransformer   
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def Embedding_Kmeans(corpus, N_clients):
    embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device = 'cuda:0') # server only

    corpus_embeddings = embedder.encode(corpus, show_progress_bar=True, batch_size=8) #smaller batch size for gpu

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

    parser.add_argument('--data_file', type=str, default='data/data_loaders/squad_1.1_data_loader.pkl',
                        metavar="DF", help='data pickle file path')

    parser.add_argument('--partition_file', type=str, default='data/partition/squad_1.1_partition.pkl',
                        metavar="PF", help='partition pickle file path')

    parser.add_argument('--embedding_file', type=str, default='data/embedding/squad_1.1_embedding.pkl',
                        metavar="EF", help='embedding pickle file path')
    
    parser.add_argument('--task_type', type=str, metavar="TT", help='task type')
    
    args = parser.parse_args()

    N_Clients = args.client_number
    data = ""
    

    with open(args.data_file, 'rb') as f:
        data = pickle.load(f)


    corpus = data['context_X']

    if args.task_type == "sequence_tagging":
        corpus = [" ".join(sentence) for sentence in corpus]


    cluster_assignment, embedding_data = Embedding_Kmeans(corpus, N_Clients)


    with open(args.embedding_file,'wb') as f:
        pickle.dump(embedding_data, f, pickle.HIGHEST_PROTOCOL)

    partition_pkl = {}
    for index, idx in enumerate(cluster_assignment):
        if idx in partition_pkl :
            partition_pkl[idx].append(index)
        else:
            partition_pkl[idx] = [index]

    print(partition_pkl[0])

    partition = ""
    with open(args.partition_file,'rb') as f:
        partition = pickle.load(f)

    partition['kmeans'] = {}
    partition['kmeans']['n_clients'] = N_Clients
    partition['kmeans']['partition_data'] = {}

    for i in sorted(partition_pkl.keys()):
        partition['kmeans']['partition_data'][i] = {}
        train, test = train_test_split(partition_pkl[i], test_size=0.4, train_size = 0.6, random_state=42)
        partition['kmeans']['partition_data'][i]['train'] = train
        partition['kmeans']['partition_data'][i]['test'] = test


    with open(args.partition_file,'wb') as f:
        pickle.dump(partition, f, pickle.HIGHEST_PROTOCOL)


