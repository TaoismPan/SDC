import utils
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from functools import partial
import numpy as np
import cPickle as cp
import os
import copy
from utils import log
import click
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
mnist = input_data.read_data_sets('data', one_hot=True)

source_name = "mnist"
num_class = 10
#print "max value: ", np.max(train_mat_data)
#print "min value: ", np.min(train_mat_data)
xs = mnist.train.images
ys = mnist.train.labels
# ys_val = ys[10000:]
# xs = xs[:10000]
# ys = ys[:10000]
xs_test = mnist.test.images
ys_test = mnist.test.labels


#xs, ys = utils.gen_balance_subset(xs, ys, num_sample=num_sample, num_class=num_class)
xs_set, ys_set = utils.split_class(xs, ys, num_class)
print "shape of ys_test: ", ys_test.shape

key_inf = []

l2_param = 0.
lr = 1e-4
batch_size = 12
num_steps = 120000

coral_param = 2e-5
#outer_intra_param = 1e-4
log_file = "log/idc_dropout_{}.txt".format(test_mat_name)
#log_file = "log/class_inadapt_{6}_coral_{0}_l2_{1}_adamlr_{2}_batchsize{3}_num{4}_num_steps{5}.txt".format(coral_param, l2_param, lr, batch_size, len(xs), num_steps, test_mat_name)

tf.set_random_seed(0)
np.random.seed(0)

with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    train_flag = tf.placeholder(tf.bool)
    d_rate = tf.placeholder("float")

with tf.name_scope('feature_generator'):
    W_conv1 = utils.weight_variable([5, 5, 1, 32], 'conv1_weight')
    b_conv1 = utils.bias_variable([32], 'conv1_bias')
    h_conv1 = tf.nn.relu(utils.conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = utils.max_pool_2x2(h_conv1)

    W_conv2 = utils.weight_variable([5, 5, 32, 64], 'conv2_weight')
    b_conv2 = utils.weight_variable([64], 'conv2_bias')
    h_conv2 = tf.nn.relu(utils.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = utils.max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    h_pool2_flat = tf.nn.dropout(h_pool2_flat, d_rate)
    W_fc1 = utils.weight_variable([7*7*64, 1024], 'fc1_weight')
    b_fc1 = utils.bias_variable([1024], 'fc1_bias')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


with tf.name_scope('slice_data'):
    whole_closs = 0
    #h_s = tf.cond(train_flag, lambda: tf.slice(h_fc1, [0, 0], [batch_size * num_class, -1]), lambda: h_fc1)
    h_s = h_fc1
    ys_true = y_
    #ys_true = tf.cond(train_flag, lambda: tf.slice(y_, [0, 0], [batch_size * num_class, -1]), lambda: y_)
    #outer_loss = utils.outer_loss(num_class, h_s, batch_size)
    for k in range(num_class):
        sk_h = tf.slice(h_fc1, [batch_size * k * 2, 0], [batch_size, -1])
        tk_h = tf.slice(h_fc1, [batch_size * (2 * k + 1), 0], [batch_size, -1])
        tmp_loss = utils.compute_coral_loss(sk_h, tk_h, coral_param, batch_size)
        whole_closs = whole_closs + tmp_loss


with tf.name_scope('classifier'):
    W_fc2 = utils.weight_variable([1024, 10], 'fc2_weight')
    b_fc2 = utils.bias_variable([10], 'fc2_bias')
    pred_logit = tf.matmul(h_s, W_fc2) + b_fc2
    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=ys_true))
    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_true, 1), tf.argmax(pred_logit, 1)), tf.float32))

all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = clf_loss + l2_loss + (whole_closs)/num_class
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
saver = tf.train.Saver(max_to_keep=20)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    xbs_set = []
    ybs_set = []
    xbt_set = []
    ybt_set = []

    for j in range(num_class):
        exec("S%s_batches = utils.batch_generator([xs_set[%s], ys_set[%s]], 2*batch_size)" % (j, j, j))
        #exec("T%s_batches = utils.batch_generator([xt_set[%s], yt_set[%s]], 2*batch_size)" % (j, j, j))
    record = 0.
    total_closs = 0.

    last_rec = 0.0
    for i in range(num_steps):
        xbs_set = []
        ybs_set = []

        for k in range(num_class):
            exec("xs%s_batch, ys%s_batch = S%s_batches.next()" % (k, k, k))
            exec("xbs_set.append(xs%s_batch)" % (k))
            exec("ybs_set.append(ys%s_batch)" % (k))

        xb_set = xbs_set
        yb_set = ybs_set
        xb = np.vstack(xb_set)
        yb = np.vstack(yb_set)
        _, s_acc, train_loss = sess.run([train_op, clf_acc, clf_loss], feed_dict={x: xb, y_: yb, train_flag: True, d_rate: 0.5})


        if i % 200 == 0:
            acc, clf_ls, closs = sess.run([clf_acc, clf_loss, whole_closs], feed_dict={x: xs_val, y_: ys_val, train_flag: False, d_rate:1.0})
            #print("Closs: ", closs*5*10000)
            acc = acc / 10.0
            clf_ls = clf_ls / 10.0
            acc_m = 0.0
            
            for k in range(10):
                tmp_acc_m, clf_ls_m = sess.run([clf_acc, clf_loss], feed_dict={x: xs_test[k*5000:(k+1)*5000], y_: ys_test[k*5000:(k+1)*5000], train_flag: False,d_rate:1.0})
                acc_m = acc_m + tmp_acc_m
            acc_m = acc_m / 10.0
            print("acc_m: ", acc_m)
            if last_rec < acc or i == 60000:
                last_rec = acc
                null_ls = []
                record = acc_m
                log("New record: {}".format(record), log_file=log_file)
                #saver.save(sess, "ckpt_sample{}/mnist_lenet".format(num_sample), global_step=i)
                #for k in range(11):
                #    train_feat  = sess.run(h_fc1, feed_dict={x: xs[5000*k : 5000*(k+1)], y_: ys[5000*k : 5000*(k+1)], train_flag:False})
                #    null_ls.append(train_feat)
                #np.save("npy/new_coral{}_train_feat.npy".format(coral_param), np.array(null_ls))
                #np.save("npy/new_coral{}_train_label.npy".format(coral_param), ys)
            log('step {}'.format(i), log_file=log_file)
            log('training loss: {}, training accuracy: {}'.format(train_loss, s_acc), log_file=log_file)
            log('test loss: {}, test accuracy: {}'.format(clf_ls_m, acc_m), log_file=log_file)
            total_closs = 0.
        if i % 1000 == 0:
            log("Current record: {}".format(record), log_file=log_file)
    log("Final rceord: {}".format(record), log_file=log_file)