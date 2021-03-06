import random
import datetime
import numpy as np
from scipy.io import mmread

LABELLED_FRACTION = 0.1
TRAIN_FRACTION = 0.6
VALIDATION_FRACTION = 0.2
TEST_FRACTION = 0.2


# Load the adjacency information from a mtx file and return it in scipy.csr_matrix format
def _load_adj_from_mtx_as_csr(mtx_file):
    g = mmread(mtx_file)
    (r, c) = g.get_shape()
    # We expect the matrix to be read in to be an adjacency matrix, hence it must be square
    assert r == c, "Error: Adjacency matrix is not square"
    return r, g.tocsr()


# Load the feature matrix
# [HACK] In reality features will be read in from a file. For now we generate and return a column of ones
def _load_features(n):
    return np.ones((n, 1))


# Load the labels map
#
# [HACK] In reality labels will be read in from a file. For now  select a random set of vertices and random 0, 1 labels
# against them
def _load_labels(n):
    num_labelled = int(n * LABELLED_FRACTION)
    labelled_vertices = random.sample(range(n), num_labelled)
    return {v: random.choice([0, 1, 1]) for v in labelled_vertices}


# Get the lists of vertices to be used as train, validation and test, given the label map
#
# [HACK] For now we just partition the labelled vertices into 3 parts. Need to figure out correct strategy,
# perhaps based on some vertex attribute
def _get_tvt_indexes(label_map):
    assert(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION == 1.0)
    key_list = list(label_map.keys())
    num_labelled = len(key_list)
    num_train = int(num_labelled * TRAIN_FRACTION)
    num_validation = int(num_labelled * VALIDATION_FRACTION)
    train_indexes = key_list[:num_train]
    validation_indexes = key_list[num_train:num_train+num_validation]
    test_indexes = key_list[num_train+num_validation:]
    return train_indexes, validation_indexes, test_indexes


# Get the lists of labels corresponding to the list of train, validation and test vertices
def _get_tvt_labels(label_map, train_indexes, validation_indexes, test_indexes):
    train_labels = [label_map[v] for v in train_indexes]
    validation_labels = [label_map[v] for v in validation_indexes]
    test_labels = [label_map[v] for v in test_indexes]
    return train_labels, validation_labels, test_labels


def load_mtx_data(adj_mtx_file):
    """ Load data from an mtx file and return the elements needed by the FastGCN algorithm

    TODO: For now only the adjacency data is loaded from the mtx file. The features and labels are mocked with
    TODO: random data. This will be addressed in subsequent updates.
    """
    print(datetime.datetime.now(), "Loading adjacency data from " + adj_mtx_file)
    (nv, adj) = _load_adj_from_mtx_as_csr(adj_mtx_file)
    print(datetime.datetime.now(), "Loading features")
    features = _load_features(nv)
    print(datetime.datetime.now(), "Loading label map")
    label_map = _load_labels(nv)
    print(datetime.datetime.now(), "Extracting TVT indexes")
    train_indexes, validation_indexes, test_indexes = _get_tvt_indexes(label_map)
    print(datetime.datetime.now(), "Extracting TVT labels")
    train_labels, validation_labels, test_labels = \
        _get_tvt_labels(label_map, train_indexes, validation_indexes, test_indexes)
    print(datetime.datetime.now(), "MTX Loading complete")
    return adj, features, train_labels, validation_labels, test_labels, train_indexes, validation_indexes, test_indexes


# Run a test with sample data
def _run_test(mtx_file):
    (adj, features,
     train_labels, validation_labels, test_labels,
     train_indexes, validation_indexes, test_indexes) = load_mtx_data(mtx_file)
    print("ADJ", adj.get_shape())
    print(adj)
    print("FEATURES", len(features))
    print(features)
    print("TVT LABELS")
    print(len(train_labels), train_labels)
    print(len(validation_labels), validation_labels)
    print(len(test_labels), test_labels)
    print("TVT INDEXES")
    print(len(train_indexes), train_indexes)
    print(len(validation_indexes), validation_indexes)
    print(len(test_indexes), test_indexes)


if __name__ == "__main__":
    LABELLED_FRACTION = 0.2
    _run_test("./data/medium-graph.mtx")
