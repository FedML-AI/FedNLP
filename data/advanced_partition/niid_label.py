import h5py
import argparse
import numpy as np
import json
from sklearn.model_selection import train_test_split
import math
from decimal import * 


getcontext().prec=128


def dynamic_batch_fill(index_marker,label_client_matrix,remaing_length,current_label_id):
    
    remaining_unfiled = remaing_length
    label_index_offset = {}
    label_remain_length_dict = {}
    total_label_remain_length = 0
    # calculate total number of all the remaing labels and each label's remaining length
    for label_id, label_list in enumerate(label_client_matrix):
        if label_id == current_label_id:
            label_remain_length_dict[label_id] = 0
            continue
        label_remaining_count = len(label_list) - index_marker[label_id]
        if label_remaining_count > 0:
            total_label_remain_length = total_label_remain_length +  label_remaining_count
        label_remain_length_dict[label_id] = label_remaining_count
        

    length_pointer = remaining_unfiled


    if total_label_remain_length > 0:
        label_sorted_by_length = {k:v for k, v in sorted(label_remain_length_dict.items(), key=lambda item: item[1])}
    else:
        label_index_offset = label_remain_length_dict
        return label_index_offset
    # for each label calculate the offset move forward by distribution of remaining labels
    for label_id in label_sorted_by_length.keys():
        fill_count = math.ceil(label_remain_length_dict[label_id] / total_label_remain_length * remaing_length)
        fill_count = min(fill_count, label_remain_length_dict[label_id])
        offset_forward = fill_count
        #if left room not enough for all offset set it to 0
        if  length_pointer - offset_forward <= 0:
            label_index_offset[label_id] = length_pointer
            length_pointer = 0
            break
        else:
            length_pointer -= offset_forward
            label_remain_length_dict[label_id] -= offset_forward
        label_index_offset[label_id] = offset_forward
    
    # still has some room unfilled
    if length_pointer > 0:
        for label_id in label_sorted_by_length.keys():
            # make sure no infinite loop happens
            fill_count = math.ceil(label_sorted_by_length[label_id] / total_label_remain_length * length_pointer)
            fill_count = min(fill_count, label_remain_length_dict[label_id])
            offset_forward = fill_count
            if  length_pointer - offset_forward <= 0:
                label_index_offset[label_id] += length_pointer
                length_pointer = 0
                break
            else:
                length_pointer -= offset_forward
                label_remain_length_dict[label_id] -= offset_forward
            label_index_offset[label_id] += offset_forward
        

    return label_index_offset




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--client_number', type=int, default='100', metavar='CN',
                        help='client number for lda partition')

    parser.add_argument('--data_file', type=str, default='data/data_files/20news_data.h5',
                        metavar="DF", help='data pickle file path')

    parser.add_argument('--partition_file', type=str, default='data/partition_files/20news_partition.h5',
                        metavar="PF", help='partition pickle file path')
    
    parser.add_argument('--task_type', type=str, metavar="TT", help='task type')

    parser.add_argument('--kmeans_num', type=int, metavar="KN", help='number of k-means cluster')

    parser.add_argument('--alpha', type=float, metavar="A", help='alpha value for LDA')
    
    args = parser.parse_args()


    print("start reading data")
    
    
    client_num = args.client_number
    alpha = args.alpha # need adjustment for each dataset

    print(alpha)
    print(isinstance(alpha,float))
    label_vocab = []
    label_assignment = np.array([])

    print('retrive data')
    # retrive total index length 
    data = h5py.File(args.data_file,"r")
    attributes = json.loads(data["attributes"][()])
    print(attributes.keys())
    total_index_list = attributes['index_list']
    test_index_list = []
    train_index_list = []
    if ("train_index_list" in attributes):
        test_index_list = attributes['test_index_list']
        print(len(test_index_list))
        train_index_list = attributes['train_index_list']
        print(len(train_index_list))

    else:
        train_length = int(len(total_index_list) * 0.9)
        train_index_list = total_index_list[0:train_length]
        test_index_list = total_index_list[train_length:]


    
    label_assignment_train = []
    label_assignment_test = []
    # retreive label vocab and label assigment the index of label assignment is the index of data assigned to this label
    # the value of each index is the label
    if args.task_type == 'text_classification':
        label_vocab = attributes['label_vocab'].keys()
        label_assignment = np.array([data['Y'][str(i)][()].decode('utf-8') for i in total_index_list])
        
        label_assignment_test = np.array([data["Y"][str(idx)][()].decode("utf-8") for idx in test_index_list])
        label_assignment_train = np.array([data["Y"][str(idx)][()].decode("utf-8") for idx in train_index_list])


    else:
        partition = h5py.File(args.partition_file,"r")
        label_vocab = [i for i in range(args.kmeans_num)]
        label_assignment = np.array(partition["kmeans_%d"%args.kmeans_num+'/client_assignment'][()])
        label_assignment_test = np.array([label_assignment[int(idx)] for idx in test_index_list])
        label_assignment_train = np.array([label_assignment[int(idx)] for idx in train_index_list])


        # convert to original test and train
        partition.close()
    
    data.close()


                
    partition_pkl_train = [[] for _ in range(client_num)]
    partition_pkl_test = [[] for _ in range(client_num)]
    
    label_client_matrix = [[] for _ in label_vocab]
    label_proportion = []
    print('start process train')
    assert len(total_index_list) == len(label_assignment)
    # shuffle indexs and calculate each label proportion of the dataset
    for index, value in enumerate(label_vocab):
        label_location = np.where( label_assignment_train == value)[0]
        label_proportion.append(len(label_location) / len(label_assignment_train))
        np.random.shuffle(label_location)
        label_client_matrix[index].extend(label_location[:])
        
    # calculate size for each train 
    index_marker = np.zeros(len(label_vocab),dtype=int)
    total_index = len(label_assignment_train)
    each_client_length = int(total_index / client_num)
    client_length = 0
    
    lable_dis = []  # debug
    client_dir_dis = np.array([alpha * l for l in label_proportion])
    print(client_dir_dis)
    
    proportions = np.random.dirichlet(client_dir_dis) # still has to be number of classes
    # add all the unused data to the client
    for client_id in range(len(partition_pkl_train)):
        assignment = partition_pkl_train[client_id]
        proportions = np.random.dirichlet(client_dir_dis) # still has to be number of classes
        client_length = min(each_client_length, total_index)
        if total_index < client_length * 2:
            client_length = total_index
        total_index -= client_length
        client_length_pointer = client_length
        temp = {}   # debug
        # for each label calculate the  offset length assigned to by Dir distribution and then  extend assignment
        for label_id, _ in enumerate(label_vocab):
            offset = round(proportions[label_id] * client_length)
            if offset >= client_length_pointer:
                offset = client_length_pointer
                client_length_pointer = 0
            else:
                if label_id == (len(label_vocab) - 1):
                    offset = client_length_pointer
                client_length_pointer -= offset
            
            temp[label_id] = offset

            start = int(index_marker[label_id])
            end = int(index_marker[label_id] + offset)
            label_total_number = len(label_client_matrix[label_id])
            # if the the label is assigned to a offset length that is more than what its remaining length
            if end > label_total_number:
                assignment.extend(label_client_matrix[label_id][start:])
                index_marker[label_id] = label_total_number

                label_index_offset = dynamic_batch_fill(index_marker,label_client_matrix,end - label_total_number,label_id)
                for fill_label_id in label_index_offset.keys():
                    start = index_marker[fill_label_id]
                    end = index_marker[fill_label_id] + label_index_offset[fill_label_id]
                    assignment.extend(label_client_matrix[fill_label_id][start:end])
                    index_marker[fill_label_id] = index_marker[fill_label_id] +  label_index_offset[fill_label_id]
            else:
                assignment.extend(label_client_matrix[label_id][start:end])
                index_marker[label_id] = index_marker[label_id] + offset

        if client_id == len(partition_pkl_train) - 1:
            print("last client fill the rest of the unfilled lable")
            for not_fillall_label_id in range(len(label_vocab)):
                if index_marker[not_fillall_label_id] < len(label_client_matrix[not_fillall_label_id]):
                    start = index_marker[not_fillall_label_id]
                    assignment.extend(label_client_matrix[not_fillall_label_id][start:])
                    index_marker[not_fillall_label_id] = len(label_client_matrix[not_fillall_label_id])
                    temp[not_fillall_label_id] += len(label_client_matrix[not_fillall_label_id][start:])




        lable_dis.append(temp)#debug


    label_client_matrix = [[] for _ in label_vocab]
    label_proportion = []
    print('start process test')
    assert len(total_index_list) == len(label_assignment)
    # shuffle indexs and calculate each label proportion of the dataset
    for index, value in enumerate(label_vocab):
        label_location = np.where( label_assignment_test == value)[0]
        label_proportion.append(len(label_location) / len(label_assignment_test))
        np.random.shuffle(label_location)
        label_client_matrix[index].extend(label_location[:])
        
        
    #calculate each size for test
    index_marker = np.zeros(len(label_vocab),dtype=int)
    total_index = len(label_assignment_test)
    each_client_length = int(total_index / client_num)
    client_length = 0
    
    lable_dis = []  # debug
    client_dir_dis = np.array([alpha * l for l in label_proportion])

    print(client_dir_dis)
    proportions = np.random.dirichlet(client_dir_dis) # still has to be number of classes

    # add all the unused data to the client
    for client_id in range(len(partition_pkl_test)):
        assignment = partition_pkl_test[client_id]
        proportions = np.random.dirichlet(client_dir_dis) # still has to be number of classes
        client_length = min(each_client_length, total_index)
        if total_index < client_length * 2:
            client_length = total_index
        total_index -= client_length
        client_length_pointer = client_length
        temp = {}   # debug
        # for each label calculate the  offset length assigned to by Dir distribution and then  extend assignment
        for label_id, _ in enumerate(label_vocab):
            offset = round(proportions[label_id] * client_length)
            if offset >= client_length_pointer:
                offset = client_length_pointer
                client_length_pointer = 0
            else:
                if label_id == (len(label_vocab) - 1):
                    offset = client_length_pointer
                client_length_pointer -= offset
            
            temp[label_id] = offset

            start = int(index_marker[label_id])
            end = int(index_marker[label_id] + offset)
            label_total_number = len(label_client_matrix[label_id])
            # if the the label is assigned to a offset length that is more than what its remaining length
            if end > label_total_number:
                assignment.extend(label_client_matrix[label_id][start:])
                index_marker[label_id] = label_total_number

                label_index_offset = dynamic_batch_fill(index_marker,label_client_matrix,end - label_total_number,label_id)
                for fill_label_id in label_index_offset.keys():
                    start = index_marker[fill_label_id]
                    end = index_marker[fill_label_id] + label_index_offset[fill_label_id]
                    assignment.extend(label_client_matrix[fill_label_id][start:end])
                    index_marker[fill_label_id] = index_marker[fill_label_id] +  label_index_offset[fill_label_id]
            else:
                assignment.extend(label_client_matrix[label_id][start:end])
                index_marker[label_id] = index_marker[label_id] + offset

        if client_id == len(partition_pkl_test) - 1:
            print("last client fill the rest of the unfilled lable")
            for not_fillall_label_id in range(len(label_vocab)):
                if index_marker[not_fillall_label_id] < len(label_client_matrix[not_fillall_label_id]):
                    start = index_marker[not_fillall_label_id]
                    assignment.extend(label_client_matrix[not_fillall_label_id][start:])
                    index_marker[not_fillall_label_id] = len(label_client_matrix[not_fillall_label_id])
                    temp[not_fillall_label_id] += len(label_client_matrix[not_fillall_label_id][start:])




        lable_dis.append(temp)#debug



    print("start dirichlet distribution")
        
    # add 
    print("store data in h5 data")
    partition = h5py.File(args.partition_file,"a")

    if args.task_type == 'text_classification':
    
        if ('/niid_label_clients=%.1f_alpha=%.1f'%(args.client_number,args.alpha) in partition):
            del partition['/niid_label_clients=%.1f_alpha=%.1f'%(args.client_number,args.alpha)]
        if ('/niid_label_clients=%df_alpha=%.1f'%(args.client_number,args.alpha) in partition):
            del partition['/niid_label_clients=%df_alpha=%.1f'%(args.client_number,args.alpha)]
        
        partition['/niid_label_clients=%d_alpha=%.1f'%(args.client_number,args.alpha) + '/n_clients'] = client_num
        partition['/niid_label_clients=%d_alpha=%.1f'%(args.client_number,args.alpha) + '/alpha'] = alpha
        for i, data in enumerate(partition_pkl_train):
            train = partition_pkl_train[i]
            test = partition_pkl_test[i]
            train_path = '/niid_label_clients=%d_alpha=%.1f'%(args.client_number,args.alpha) + '/partition_data/'+str(i)+'/train/'
            test_path = '/niid_label_clients=%d_alpha=%.1f'%(args.client_number,args.alpha) + '/partition_data/'+str(i)+'/test/'
            partition[train_path] = train
            partition[test_path] = test
        partition.close()

    else:

        if('/niid_cluster_clients=%.1f_alpha=%.1f'%(args.client_number,args.alpha) in partition):
            del partition['/niid_cluster_clients=%.1f_alpha=%.1f'%(args.client_number,args.alpha)]

        if('/niid_cluster_clients=%d_alpha=%d'%(args.client_number,args.alpha) in partition):
            del partition['/niid_cluster_clients=%.1f_alpha=%.1f'%(args.client_number,args.alpha)]
    
        partition['/niid_cluster_clients=%d_alpha=%.1f'%(args.client_number,args.alpha)+ '/n_clients'] = client_num
        partition['/niid_cluster_clients=%d_alpha=%.1f'%(args.client_number,args.alpha) + '/alpha'] = alpha
        for i, data in enumerate(partition_pkl_train):
            train = partition_pkl_train[i]
            test = partition_pkl_test[i]            
            train_path ='/niid_cluster_clients=%d_alpha=%.1f'%(args.client_number,args.alpha)+ '/partition_data/'+str(i)+'/train/'
            test_path = '/niid_cluster_clients=%d_alpha=%.1f'%(args.client_number,args.alpha) + '/partition_data/'+str(i)+'/test/'
            partition[train_path] = train
            partition[test_path] = test
        partition.close()

main()


