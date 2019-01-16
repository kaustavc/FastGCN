from __future__ import division
from __future__ import print_function

import time
import datetime
import tensorflow as tf
import scipy.sparse as sp

from mtx_loader import load_mtx_data
from utils import *
from models import GCN_APPRO_Mix
import json
from networkx.readwrite import json_graph
import os
import sys

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_mix', 'Model string.')  # 'gcn', 'gcn_appr'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


def transferLabel2Onehot(labels, N):
    y = np.zeros((len(labels),N))
    for i in range(len(labels)):
        pos = labels[i]
        y[i,pos] =1
    return y


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]


def construct_feeddict_forMixlayers(AXfeatures, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['AXfeatures']: AXfeatures})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: AXfeatures[1].shape})
    return feed_dict


def train(adj_file, rank1=None):
    print(datetime.datetime.now(), "Entered train() with adj_file = " + adj_file)
    # config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
    #                 inter_op_parallelism_threads = 1,
    #                 intra_op_parallelism_threads = 4,
    #                 log_device_placement=False)
    adj, features, y_train, y_val, y_test,train_index, val_index, test_index = load_mtx_data(adj_file)
    adj = adj+adj.T

    print(datetime.datetime.now(), "FastGCN: Checkpoint 1")
    y_train = transferLabel2Onehot(y_train, 2)
    y_val = transferLabel2Onehot(y_val, 2)
    y_test = transferLabel2Onehot(y_test, 2)

    # OPT: Can we eliminating this? Needed later else todense() fails below
    features = sp.lil_matrix(features)

    adj_train = adj[train_index, :][:, train_index]


    numNode_train = adj_train.shape[0]


    # print("numNode", numNode)



    if FLAGS.model == 'gcn_mix':
        normADJ_train = nontuple_preprocess_adj(adj_train)
        normADJ = nontuple_preprocess_adj(adj)
        # normADJ_val = nontuple_preprocess_adj(adj_val)
        # normADJ_test = nontuple_preprocess_adj(adj_test)

        num_supports = 2
        model_func = GCN_APPRO_Mix
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Some preprocessing
    print(datetime.datetime.now(), "FastGCN: Checkpoint 2")
    features = nontuple_preprocess_features(features).todense()

    train_features = normADJ_train.dot(features[train_index])
    features = normADJ.dot(features)
    nonzero_feature_number = len(np.nonzero(features)[0])
    nonzero_feature_number_train = len(np.nonzero(train_features)[0])


    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32) ,
        'AXfeatures': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features.shape[-1], logging=True)

    # Initialize session
    # sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)

    # Define model evaluation function
    def evaluate(features, support, labels, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feeddict_forMixlayers(features, support, labels, placeholders)
        print(datetime.datetime.now(), "Starting sess.run for evaluate")
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val,
                            options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    cost_val = []

    p0 = column_prop(normADJ_train)

    # testSupport = [sparse_to_tuple(normADJ), sparse_to_tuple(normADJ)]
    valSupport = sparse_to_tuple(normADJ[val_index, :])
    testSupport = sparse_to_tuple(normADJ[test_index, :])

    print(datetime.datetime.now(), "FastGCN: Checkpoint 3")
    all_epochs_start_time = time.time()
    maxACC = 0.0
    # Train model
    for epoch in range(FLAGS.epochs):
        print(datetime.datetime.now(), "Starting training epoch %d" % epoch)

        current_epoch_start_time = time.time()
        n = 0
        for batch in iterate_minibatches_listinputs([normADJ_train, y_train], batchsize=256, shuffle=True):
            current_batch_start_time = time.time()
            if (n % 1000) == 0:
                print(datetime.datetime.now(), "Starting training epoch/batch : %d/%d " % (epoch, n))

            [normADJ_batch, y_train_batch] = batch

            # p1 = column_prop(normADJ_batch)
            if rank1 is None:
                support1 = sparse_to_tuple(normADJ_batch)
                features_inputs = train_features
            else:
                distr = np.nonzero(np.sum(normADJ_batch, axis=0))[1]
                if rank1 > len(distr):
                    q1 = distr
                else:
                    q1 = np.random.choice(distr, rank1, replace=False, p=p0[distr]/sum(p0[distr]))  # top layer

                # q1 = np.random.choice(np.arange(numNode_train), rank1, p=p0)  # top layer

                support1 = sparse_to_tuple(normADJ_batch[:, q1].dot(sp.diags(1.0 / (p0[q1] * rank1))))
                if len(support1[1])==0:
                    continue

                features_inputs = train_features[q1, :]  # selected nodes for approximation
            # Construct feed dictionary
            feed_dict = construct_feeddict_forMixlayers(features_inputs, support1, y_train_batch,
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            # print(datetime.datetime.now(), "Starting sess.run for batch")
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict,
                            options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
            n = n+1
            if (n % 1000) == 0:
                print(datetime.datetime.now(), "Finished epoch/batch %d/%d after %d seconds" % (epoch, n, time.time() - current_epoch_start_time))


        # Validation
        print(datetime.datetime.now(), "Starting validation for epoch %d" % epoch)
        cost, acc, duration = 0, 0, 0 #evaluate(features, valSupport, y_val,  placeholders)
        cost_val.append(cost)
        print(datetime.datetime.now(), "Done validation for epoch %d" % epoch)

        if epoch > 20 and acc>maxACC:
            maxACC = acc
            saver.save(sess, "tmp/tmp_MixModel_sampleA_full.ckpt")

        # Print results
        print(datetime.datetime.now(), "Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time per batch=", "{:.5f}".format((time.time() - current_epoch_start_time)/n))

        if epoch%5==0:
            # Validation
            test_cost, test_acc, test_duration = 0, 0, 0 #evaluate(features, testSupport, y_test, placeholders)
            print(datetime.datetime.now(), "training time by far=", "{:.5f}".format(time.time() - all_epochs_start_time),
                  "epoch = {}".format(epoch + 1),
                  "cost=", "{:.5f}".format(test_cost),
                  "accuracy=", "{:.5f}".format(test_acc))

        if epoch > FLAGS.early_stopping and np.mean(cost_val[-2:]) > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            # print("Early stopping...")
            break

    train_duration = time.time() - all_epochs_start_time
    # Testing
    if os.path.exists("tmp/tmp_MixModel_sampleA_full.ckpt.index"):
        saver.restore(sess, "tmp/tmp_MixModel_sampleA_full.ckpt")
    test_cost, test_acc, test_duration = 0, 0, 0 # evaluate(features, testSupport, y_test, placeholders)
    print(datetime.datetime.now(), "rank1 = {}".format(rank1), "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "training time=", "{:.5f}".format(train_duration),
          "epoch = {}".format(epoch+1),
          "test time=", "{:.5f}".format(test_duration))




def test(adj_file, rank1=None):
    # config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
    #                 inter_op_parallelism_threads = 1,
    #                 intra_op_parallelism_threads = 4,
    #                 log_device_placement=False)
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = load_mtx_data(adj_file)
    adj = adj + adj.T

    y_train = transferLabel2Onehot(y_train, 2)
    y_test = transferLabel2Onehot(y_test, 2)

    features = sp.lil_matrix(features)


    numNode_train = y_train.shape[0]

    # print("numNode", numNode)



    if FLAGS.model == 'gcn_mix':
        normADJ = nontuple_preprocess_adj(adj)
        normADJ_test = normADJ[test_index, :]
        # normADJ_val = nontuple_preprocess_adj(adj_val)
        # normADJ_test = nontuple_preprocess_adj(adj_test)

        num_supports = 2
        model_func = GCN_APPRO_Mix
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Some preprocessing
    features = nontuple_preprocess_features(features).todense()

    features = normADJ.dot(features)


    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32),
        'AXfeatures': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features.shape[-1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feeddict_forMixlayers(features, support, labels, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, "tmp/tmp_MixModel_sampleA.ckpt")

    cost_val = []

    p0 = column_prop(normADJ_test)


    t = time.time()

    if rank1 is None:
        support1 = sparse_to_tuple(normADJ_test)
        features_inputs = features
    else:
        distr = np.nonzero(np.sum(normADJ_test, axis=0))[1]
        if rank1 > len(distr):
            q1 = distr
        else:
            q1 = np.random.choice(distr, rank1, replace=False, p=p0[distr] / sum(p0[distr]))  # top layer

        # q1 = np.random.choice(np.arange(numNode_train), rank1, p=p0)  # top layer

        support1 = sparse_to_tuple(normADJ_test[:, q1].dot(sp.diags(1.0 / (p0[q1] * rank1))))


        features_inputs = features[q1, :]  # selected nodes for approximation

    test_cost, test_acc, test_duration = evaluate(features_inputs, support1, y_test,
                                                  placeholders)


    test_duration = time.time() - t
    print("rank1 = {}".format(rank1), "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "test time=", "{:.5f}".format(test_duration))

if __name__=="__main__":
    # main(None)
    print("Starting fastGCN Training on " + sys.argv[1])
    train(sys.argv[1])
    # for k in [25, 50, 100, 200, 400]:
    #     main(k)