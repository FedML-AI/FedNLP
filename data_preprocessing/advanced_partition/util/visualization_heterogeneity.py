import os
import h5py
import matplotlib.pyplot as plt
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--partition_name', type=str, metavar='PN',
                        help='name of the method ')
parser.add_argument('--partition_file', type=str, default='data/partition_files/wikiner_partition.h5',
                        metavar="PF", help='data partition path')
parser.add_argument('--data_file', type=str, default='data/data_files/wikiner_data.h5',
                        metavar="DF", help='data file path')
parser.add_argument('--task_name', type=str, metavar="TN", help="task name")

parser.add_argument('--figure_path', type=str, metavar="TN", help="the place to store generated figures")

parser.add_argument('--task_type', type=str, default='name entity recognition',metavar="TT", help="task type")


args = parser.parse_args()


temp = "kmeans"
client_assignment = []

if args.task_type == "text_classification":
    data = h5py.File(args.data_file,"r")
    client_assignment = [data['Y'][i][()] for i in data['Y'].keys()]
    for index, value in enumerate(set(client_assignment)):
        client_assignment = [index if i == value else i for i in client_assignment ]
    data.close()

else:
    f = h5py.File(args.partition_file,"r")
    for i in f.keys():
        if temp in i :
            client_assignment = f[i+"/client_assignment/"][()]
            break
    f.close()
print(set(client_assignment))
cluster_list = list(set(client_assignment))
class_list = ["c_" + str(i) for i in cluster_list]


partition_data_path = "/"+args.partition_name+"/partition_data/"

#randomly sample 4 clients 
clients = [1,2,3,4]
f = h5py.File(args.partition_file,"r")

for i in clients:

    client_sample = []
    cluster_sample = []
    client_sample.extend(f[partition_data_path+str(i)+'/train/'][()])
    client_sample.extend(f[partition_data_path+str(i)+'/test/'][()])
    cluster_sample = [client_assignment[i] for i in client_sample]
    cluster_figure = [cluster_sample.count(i) for i in cluster_list]

    data_dir = args.figure_path
    fig = plt.figure(figsize=(len(cluster_list)+5, 10))
    ax = fig.gca()
    y_pos = [i for i in cluster_list]
    ax.bar(y_pos,cluster_figure,width=0.5, align='center')
    plt.xticks(y_pos,class_list)
    ax.plot(cluster_figure,linewidth=2.0,color='r')
    fig_name = args.task_name + "_client_" + str(i) + "_%s_actual geterigeneous data distribution.png" % args.partition_name
    fig_dir = os.path.join(data_dir, fig_name)
    plt.title(args.task_name + "_client_" + str(i) +  "_data_distribution")
    plt.xlabel('class idx')
    plt.ylabel("sample number")
    plt.savefig(fig_dir)
    plt.close()

f.close()
