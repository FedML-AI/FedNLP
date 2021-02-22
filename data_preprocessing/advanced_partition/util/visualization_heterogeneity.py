import os
import h5py
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--partition_name', type=str, metavar='PN',
                        help='name of the method ')
parser.add_argument('--partition_file', type=str, default='data/partition_files/wikiner_partition.h5',
                        metavar="DF", help='data file path')
parser.add_argument('--task_name', type=str, metavar="TN", help="task name")

parser.add_argument('--figure_path', type=str, metavar="TN", help="the place to store generated figures")

args = parser.parse_args()

f = h5py.File(args.partition_file,"r")

temp = "kmeans"
client_assignment = []
for i in f.keys():
    if temp in i :
        client_assignment = f[i+"/client_assignment/"][()]
        break

partition_data_path = "/"+args.partition_name+"/partition_data/"

cluster_list = list(set(client_assignment))
cluster_lists = ["c" + str(i) for i in cluster_list]

#randomly sample 4 clients 
clients = [1,2,3,4]

for i in clients:

    client_sample = []
    cluster_sample = []
    client_sample.extend(f[partition_data_path+str(i)+'/train/'][()])
    client_sample.extend(f[partition_data_path+str(i)+'/test/'][()])
    cluster_sample = [client_assignment[i] for i in client_sample]
    cluster_figure = [cluster_sample.count(i) for i in cluster_list]

    data_dir = args.figure_path
    fig = plt.figure()
    ax = fig.gca()
    ax.bar(cluster_lists,cluster_figure)
    ax.plot(cluster_figure,linewidth=2.0,color='r')
    fig_name = args.task_name + " client " + str(i) + " %s_actual geterigeneous data distribution.png" % args.partition_name
    fig_dir = os.path.join(data_dir, fig_name)
    plt.title(args.task_name + " client " + str(i) +  " data distribution")
    plt.xlabel('class idx')
    plt.ylabel("sample number")
    plt.savefig(fig_dir)
    plt.close()

f.close()
