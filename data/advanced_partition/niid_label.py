import h5py
import argparse
import numpy as np
import json
from sklearn.model_selection import train_test_split
import math
from decimal import *


def dynamic_batch_fill(label_index_tracker, label_index_matrix, remaining_length,
                       current_label_id):
    """
    params
    ------------------------------------------------------------------------
    label_index_tracker : 1d numpy array track how many data each label has used 
    label_index_matrix : 2d array list of indexs of each label
    remaining_length : int remaining empty space in current partition client list
    current_label_id : int current round label id
    ------------------------------------------------------------------------

    return 
    ---------------------------------------------------------
    label_index_offset: dict  dictionary key is label id 
    and value is the offset associated with this key
    ----------------------------------------------------------
    """
    remaining_unfiled = remaining_length
    label_index_offset = {}
    label_remain_length_dict = {}
    total_label_remain_length = 0
    # calculate total number of all the remaing labels and each label's remaining length
    for label_id, label_list in enumerate(label_index_matrix):
        if label_id == current_label_id:
            label_remain_length_dict[label_id] = 0
            continue
        label_remaining_count = len(label_list) - label_index_tracker[label_id]
        if label_remaining_count > 0:
            total_label_remain_length = (total_label_remain_length +
                                         label_remaining_count)
        label_remain_length_dict[label_id] = label_remaining_count
    length_pointer = remaining_unfiled

    if total_label_remain_length > 0:
        label_sorted_by_length = {k: v for k, v in sorted(label_remain_length_dict.items(),
                               key=lambda item: item[1]) }
    else:
        label_index_offset = label_remain_length_dict
        return label_index_offset
    # for each label calculate the offset move forward by distribution of remaining labels
    for label_id in label_sorted_by_length.keys():
        fill_count = math.ceil(label_remain_length_dict[label_id] /
                               total_label_remain_length * remaining_length)
        fill_count = min(fill_count, label_remain_length_dict[label_id])
        offset_forward = fill_count
        # if left room not enough for all offset set it to 0
        if length_pointer - offset_forward <= 0:
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
            fill_count = math.ceil(label_sorted_by_length[label_id] /
                                   total_label_remain_length * length_pointer)
            fill_count = min(fill_count, label_remain_length_dict[label_id])
            offset_forward = fill_count
            if length_pointer - offset_forward <= 0:
                label_index_offset[label_id] += length_pointer
                length_pointer = 0
                break
            else:
                length_pointer -= offset_forward
                label_remain_length_dict[label_id] -= offset_forward
            label_index_offset[label_id] += offset_forward

    return label_index_offset


def label_skew_process(label_vocab, label_assignment, client_num, alpha):
    """
    params
    -------------------------------------------------------------------
    label_vocab : dict label vocabulary of the dataset 
    label_assignment : 1d list a list of label, the index of list is the index associated to label
    client_num : int number of clients
    alpha : float similarity of each client, the larger the alpha the similar data for each client
    -------------------------------------------------------------------
    return 
    ------------------------------------------------------------------
    partition_result : 2d array list of partition index of each client 
    ------------------------------------------------------------------
    """
    label_index_matrix = [[] for _ in label_vocab]
    label_proportion = []
    partition_result = [[] for _ in range(client_num)]
    client_length = 0
    # shuffle indexs and calculate each label proportion of the dataset
    for index, value in enumerate(label_vocab):
        label_location = np.where(label_assignment == value)[0]
        label_proportion.append(len(label_location) / len(label_assignment))
        np.random.shuffle(label_location)
        label_index_matrix[index].extend(label_location[:])

    # calculate size for each partition client
    label_index_tracker = np.zeros(len(label_vocab), dtype=int)
    total_index = len(label_assignment)
    each_client_index_length = int(total_index / client_num)
    client_dir_dis = np.array([alpha * l for l in label_proportion])
    print(client_dir_dis)

    proportions = np.random.dirichlet(client_dir_dis)
    # add all the unused data to the client
    for client_id in range(len(partition_result)):
        each_client_partition_result = partition_result[client_id]
        proportions = np.random.dirichlet(client_dir_dis)
        client_length = min(each_client_index_length, total_index)
        if total_index < client_length * 2:
            client_length = total_index
        total_index -= client_length
        client_length_pointer = client_length
        # for each label calculate the offset length assigned to by Dir distribution and then extend assignment
        for label_id, _ in enumerate(label_vocab):
            offset = round(proportions[label_id] * client_length)
            if offset >= client_length_pointer:
                offset = client_length_pointer
                client_length_pointer = 0
            else:
                if label_id == (len(label_vocab) - 1):
                    offset = client_length_pointer
                client_length_pointer -= offset

            start = int(label_index_tracker[label_id])
            end = int(label_index_tracker[label_id] + offset)
            label_data_length = len(label_index_matrix[label_id])
            # if the the label is assigned to a offset length that is more than what its remaining length
            if end > label_data_length:
                each_client_partition_result.extend(label_index_matrix[label_id][start:])
                label_index_tracker[label_id] = label_data_length
                label_index_offset = dynamic_batch_fill( label_index_tracker, label_index_matrix,
                    end - label_data_length, label_id)
                for fill_label_id in label_index_offset.keys():
                    start = label_index_tracker[fill_label_id]
                    end = (label_index_tracker[fill_label_id] +
                           label_index_offset[fill_label_id])
                    each_client_partition_result.extend(
                        label_index_matrix[fill_label_id][start:end])
                    label_index_tracker[fill_label_id] = (label_index_tracker[fill_label_id] 
                                                            + label_index_offset[fill_label_id])
            else:
                each_client_partition_result.extend(label_index_matrix[label_id][start:end])
                label_index_tracker[label_id] = label_index_tracker[label_id] + offset
        # if last client still has empty rooms, fill empty rooms with the rest of the unused data
        if client_id == len(partition_result) - 1:
            print("Last client fill the rest of the unfilled lables.")
            for not_fillall_label_id in range(len(label_vocab)):
                if label_index_tracker[not_fillall_label_id] < len(label_index_matrix[not_fillall_label_id]):
                    start = label_index_tracker[not_fillall_label_id]
                    each_client_partition_result.extend(
                        label_index_matrix[not_fillall_label_id][start:])
                    label_index_tracker[not_fillall_label_id] = len(
                        label_index_matrix[not_fillall_label_id])
        partition_result[client_id] = each_client_partition_result
        return partition_result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client_number",
        type=int,
        default="100",
        metavar="CN",
        help="client number for lda partition",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="data/data_files/20news_data.h5",
        metavar="DF",
        help="data pickle file path",
    )

    parser.add_argument(
        "--partition_file",
        type=str,
        default="data/partition_files/20news_partition.h5",
        metavar="PF",
        help="partition pickle file path",
    )

    parser.add_argument("--task_type",
                        type=str,
                        metavar="TT",
                        help="task type: []")

    parser.add_argument("--skew_type",
                        type=str,
                        metavar="TT",
                        help="skeq type: [label, feature]")

    parser.add_argument("--kmeans_num",
                        type=int,
                        metavar="KN",
                        help="number of k-means cluster")

    parser.add_argument("--alpha",
                        type=float,
                        metavar="A",
                        help="alpha value for LDA")

    args = parser.parse_args()

    # TODO: add a random seed arg  
    # np.random.seed(args.seed)


    print("start reading data")
    client_num = args.client_number
    alpha = args.alpha  # need adjustment for each dataset
    label_vocab = []
    label_assignment = np.array([])

    print("retrieve data")
    # retrieve total index length
    data = h5py.File(args.data_file, "r")
    attributes = json.loads(data["attributes"][()])
    print(attributes.keys())
    total_index_list = attributes["index_list"]
    test_index_list = []
    train_index_list = []

    if "train_index_list" in attributes:
        test_index_list = attributes["test_index_list"]
        print(len(test_index_list))
        train_index_list = attributes["train_index_list"]
        print(len(train_index_list))
    else:
        # some dataset like wikiner do not have presplited train test dataset so we split the data
        train_length = int(len(total_index_list) * 0.9)
        train_index_list = total_index_list[0:train_length]
        test_index_list = total_index_list[train_length:]

    label_assignment_train = []
    label_assignment_test = []
    # retreive label vocab and label assigment the index of label assignment is the index of data assigned to this label
    # the value of each index is the label
    # label assignment's index all the index of the data and the label_assignment[index] stands for the label correspond to that index
    if args.skew_type == "label":
        if args.task_type == "text_classification":
            label_vocab = attributes["label_vocab"].keys()
            label_assignment = np.array(
                [data["Y"][str(i)][()].decode("utf-8") for i in total_index_list])
            label_assignment_test = np.array([
                data["Y"][str(idx)][()].decode("utf-8") for idx in test_index_list
            ])
            label_assignment_train = np.array([
                data["Y"][str(idx)][()].decode("utf-8") for idx in train_index_list
            ])
        elif args.task_type == "sequence_tagging":
            # TODO: convert seq of tags --> a str for the sorted set of tags 
            # e.g.,  "OOOO B-PER I-PER  OOO B-LOC OOO B-LOC " ---> set{B-PER, B-LOC} --sorted--> "LOC-PER"
            # print(len(differnt types of pseudo-label))
            pass
        else:
            print("Not Implemented.")
            exit()
    elif args.skew_type == "feature":
        # input feature skew --> Kmeans clustering + dir.
        partition = h5py.File(args.partition_file, "r")
        label_vocab = [i for i in range(args.kmeans_num)]
        label_assignment = np.array(partition["kmeans_%d" % args.kmeans_num +
                                              "/client_assignment"][()])
        label_assignment_test = np.array(
            [label_assignment[int(idx)] for idx in test_index_list])
        label_assignment_train = np.array(
            [label_assignment[int(idx)] for idx in train_index_list])
        partition.close()

    data.close()

    assert len(total_index_list) == len(label_assignment)
    print("start train data processing")
    partition_result_train = label_skew_process(label_vocab,
                                                label_assignment_train,
                                                client_num, alpha)
    print("start test data processing")
    partition_result_test = label_skew_process(label_vocab,
                                               label_assignment_test,
                                               client_num, alpha)

    print("store data in h5 data")
    partition = h5py.File(args.partition_file, "a")

    flag_str = "label" if args.skew_type == "label" else "cluster" 
    # delete the old partition files in h5 so that we can write to  the h5 file
    if ("/niid"+flag_str+"clients=%.1f_alpha=%.1f" %
        (args.client_number, args.alpha) in partition):
        del partition["/niid"+flag_str+"clients=%.1f_alpha=%.1f" %
                        (args.client_number, args.alpha)]
    if ("/niid"+flag_str+"clients=%df_alpha=%.1f" %
        (args.client_number, args.alpha) in partition):
        del partition["/niid"+flag_str+"clients=%df_alpha=%.1f" %
                        (args.client_number, args.alpha)]

    partition["/niid"+flag_str+"clients=%d_alpha=%.1f" %
                (args.client_number, args.alpha) + "/n_clients"] = client_num
    partition["/niid"+flag_str+"clients=%d_alpha=%.1f" %
                (args.client_number, args.alpha) + "/alpha"] = alpha
    for partition_id in range(client_num):
        train = partition_result_train[partition_id]
        test = partition_result_test[partition_id]
        train_path = ("/niid"+flag_str+"clients=%d_alpha=%.1f" %
                        (args.client_number, args.alpha) +
                        "/partition_data/" + str(partition_id) + "/train/")
        test_path = ("/niid"+flag_str+"clients=%d_alpha=%.1f" %
                        (args.client_number, args.alpha) +
                        "/partition_data/" + str(partition_id) + "/test/")
        partition[train_path] = train
        partition[test_path] = test
    partition.close()

main()
