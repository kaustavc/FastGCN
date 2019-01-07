import random
import numpy as np
from scipy.io import mmread

LABELLED_FRACTION = 0.2
TRAIN_FRACTION=0.6
VALIDATION_FRACTION=0.2
TEST_FRACTION=0.2

"""
Read the adjacency information as a matrix market matrix and return it in scipy.csr_matrix format
"""
def read_adj_from_mtx_as_csr(mtx_file):
    g = mmread(mtx_file)
    (r, c) = g.get_shape()
    # We expect the matrix to be read in to be an adjacency matrix, hence it must be square
    assert r == c, "Error: Adjacency matrix is not square"
    return r, g.tocsr()


"""
Get the feature matrix

HACK: In reality features will be read in from a file. For now we generate and return a column of ones
"""
def get_feature_matrix(n):
    return np.ones((n, 1), dtype='int32')


"""
Get the labels map

HACK: In reality labels will be read in from a file. For now we select a random set of vertices and random 0, 1 labels
against them
"""
def get_labels(n):
    num_labelled = int(n * LABELLED_FRACTION)
    labelled_vertices = random.sample(range(n), num_labelled)
    return { v:random.choice([0, 1, 1]) for v in labelled_vertices }

"""
Get the lists of vertices to be used as train, validation and test, given the label map
"""
def get_tvt_indexes(label_map):
    assert(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION == 1.0)
    key_list = list(label_map.keys())
    num_labelled = len(key_list)
    num_train = int(num_labelled * TRAIN_FRACTION)
    num_validation = int(num_labelled * VALIDATION_FRACTION)
    train_indexes = key_list[:num_train]
    validation_indexes = key_list[num_train:num_train+num_validation]
    test_indexes = key_list[num_train+num_validation:]
    return train_indexes, validation_indexes, test_indexes


"""
Get the lists of labels corresponding to the list of train, validation and test vertices
"""
def get_tvt_labels(label_map, train_indexes, validation_indexes, test_indexes):
    train_labels = [label_map[v] for v in train_indexes]
    validation_labels = [label_map[v] for v in validation_indexes]
    test_labels = [label_map[v] for v in test_indexes]
    return train_labels, validation_labels, test_labels


def load_fk_data(adj_mtx_file):
    # Get the adjacency matrix
    (nv, adj) = read_adj_from_mtx_as_csr(adj_mtx_file)
    features = get_feature_matrix(nv)
    label_map = get_labels(nv)
    train_indexes, validation_indexes, test_indexes = get_tvt_indexes(label_map)
    train_labels, validation_labels, test_labels = get_tvt_labels(label_map, train_indexes, validation_indexes, test_indexes)
    return adj, features, train_labels, validation_labels, test_labels, train_indexes, validation_indexes, test_indexes


if __name__ == "__main__":
    LABELLED_FRACTION = 0.5
    file = "./data/jgl009.mtx"
    adj, features, train_labels, validation_labels, test_labels, train_indexes, validation_indexes, test_indexes = load_fk_data(file)
    print("ADJ")
    print(adj)
    print("FEATURES")
    print(features)
    print("TVT LABELS")
    print(train_labels)
    print(validation_labels)
    print(test_labels)
    print("TVT INDEXES")
    print(train_indexes)
    print(validation_indexes)
    print(test_indexes)
