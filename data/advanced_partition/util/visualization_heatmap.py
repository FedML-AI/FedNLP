import os
import h5py
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance

parser = argparse.ArgumentParser()

parser.add_argument('--partition_name', type=str, metavar='PN',
                        help='name of the method ')
parser.add_argument('--partition_file', type=str, default='data/partition_files/wikiner_partition.h5',
                        metavar="PF", help='data partition path')
parser.add_argument('--data_file', type=str, default='data/data_files/wikiner_data.h5',
                        metavar="DF", help='data file path')
parser.add_argument('--task_name', type=str, metavar="TN", help="task name")

parser.add_argument('--client_number', type=int, metavar="CN", help="client number of this partition method")

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
partition_data_path = "/"+args.partition_name+"/partition_data/"

client_numbers = args.client_number
client_index = list(range(client_numbers))
print(client_index)
client_data_distribution = []

f = h5py.File(args.partition_file,"r")
for i in client_index:
    temp = []
    single_client_data = []
    temp.extend(f[partition_data_path+str(i)+'/train/'][()])
    temp.extend(f[partition_data_path+str(i)+'/test/'][()])
    single_client_data = np.array([client_assignment[i] for i in temp])
    client_data_distribution.append(single_client_data)
f.close()
heat_map_data = np.zeros((client_numbers,client_numbers))

for i in range(client_numbers):
    for j in range(client_numbers):
        offset = abs(len(client_data_distribution[i]) - len(client_data_distribution[j]))
        if len(client_data_distribution[i]) < len(client_data_distribution[j]):
            client_data_distribution[i] = np.pad(client_data_distribution[i],(0,offset))
        if len(client_data_distribution[i]) > len(client_data_distribution[j]):
            client_data_distribution[j] = np.pad(client_data_distribution[j],(0,offset))
        heat_map_data[i][j] = distance.jensenshannon(client_data_distribution[i],client_data_distribution[j])

#reorder index based on the sum of distance in each client
client_data_distribution_reorder_index = [np.where(heat_map_data == i)[0][0] for i in sorted(heat_map_data, key=lambda client: sum(client), reverse=True)]
client_data_distribution_reorder = []
client_sum_order = sorted([sum(i) for i in heat_map_data], reverse=True)
#reorder the matrix based on the reorder index
for index, value in enumerate(heat_map_data):
    heat_map_data[index] = value[client_data_distribution_reorder_index]
heat_map_data = heat_map_data[client_data_distribution_reorder_index]

data_dir = args.figure_path
fig_name = args.task_name + "_%s_clients_heatmap.png" % args.partition_name
fig_dir = os.path.join(data_dir, fig_name)
fig_dims = (30, 22)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(heat_map_data,linewidths = 0.05, cmap="Blues")
plt.title(args.task_name + "_%s_clients_heatmap" % args.partition_name)
plt.savefig(fig_dir)

plt.figure(figsize=(20,15))
sns.distplot(client_sum_order)
fig_name = args.task_name + "_%s_clients_sum_distplot.png" % args.partition_name
fig_dir = os.path.join(data_dir, fig_name)
plt.title(args.task_name + "_%s_clients_sum_distplot" % args.partition_name)
plt.savefig(fig_dir,bbox_inches='tight')