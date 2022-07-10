import gzip
import cPickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
import copy
import random
from functools import partial

def add_gauss_noise(imgs, var=0.01):
    mean = 0.
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, imgs.shape)
    noisy = imgs + gauss
    noisy = np.clip(noisy, 0.0, 1.0)

    return noisy


def group_id_2_label(group_ids, num_class):
    labels = np.zeros([len(group_ids), num_class])
    for i in range(len(group_ids)):
        labels[i, int(group_ids[i])] = 1
    return labels


def load_usps(data_dir, one_hot=True, flatten=True):
    usps = pkl.load(gzip.open(data_dir, "rb"))
    # 7438, 1, 28, 28
    train_images = usps[0][0]
    # 7438x[0~9]
    train_labels = usps[0][1]
    # 1860
    test_images = usps[1][0]
    test_labels = usps[1][1]
    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
    if one_hot:
        train_labels = group_id_2_label(train_labels, 10)
        test_labels = group_id_2_label(test_labels, 10)
    return train_images, train_labels, test_images, test_labels




def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, input_type='dense'):
    with tf.name_scope(layer_name):
        weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1. / tf.sqrt(input_dim / 2.)), name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')
        if input_type == 'sparse':
            activations = act(tf.sparse_tensor_dense_matmul(input_tensor, weight) + bias)
        else:
            activations = act(tf.matmul(input_tensor, weight) + bias)
        return activations


def mmd_loss(xs, xt, mmd_param, batch_size):
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    loss_value = maximum_mean_discrepancy(xs, xt, kernel=gaussian_kernel)
    mmd_loss = mmd_param * tf.maximum(1e-4, loss_value)

    return mmd_loss


def split_class(xt, yt, num_class=10):
    xt_set = []
    yt_set = []
    yt_tmp = copy.deepcopy(yt)
    if len(yt.shape) > 1:
        yt_tmp = np.argmax(yt, axis=1)
    for class_order in range(num_class):
        classi_index = np.where(yt_tmp == class_order)[0]
        xt_set.append(xt[classi_index])
        yt_set.append(yt[classi_index])
    return xt_set, yt_set


def compute_coral_loss(h_s, h_t, coral_param, b_size):
    batch_size = tf.cast(b_size, tf.float32)
    _D_s = tf.reduce_sum(h_s, axis=0, keep_dims=True)
    _D_t = tf.reduce_sum(h_t, axis=0, keep_dims=True)
    C_s = (tf.matmul(tf.transpose(h_s), h_s) - tf.matmul(tf.transpose(_D_s), _D_s) / (batch_size)) / (batch_size - 1)
    C_t = (tf.matmul(tf.transpose(h_t), h_t) - tf.matmul(tf.transpose(_D_t), _D_t) / (batch_size)) / (batch_size - 1)
    coral_loss = coral_param * tf.nn.l2_loss(C_s - C_t)

    return coral_loss


def gen_balance_subset(xs, ys, num_sample=1000, num_class=10):
    single_class_num = num_sample / num_class
    samples_index = []
    ys_unonehot = np.argmax(ys, axis=1)
    for class_key in range(num_class):
        tmp_ls = np.where(ys_unonehot == class_key)[0].tolist()
        samples_index = samples_index + tmp_ls[:single_class_num]
    random.shuffle(samples_index)

    return xs[samples_index], ys[samples_index]


def log(text, log_file):
    print(text)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(text + '\n')
            f.flush()
            f.close()

def mix_set(xset1, yset1, xset2, yset2, num_class=10):
    x_mix = []; y_mix = []
    for cls_idx in range(num_class):
        x_mix.append(xset1[cls_idx])
        x_mix.append(xset2[cls_idx])
        y_mix.append(yset1[cls_idx])
        y_mix.append(yset2[cls_idx])
    return x_mix, y_mix

def center_loss(xs, batch_size, center_param, num_class=10):
    total_loss = 0.0
    for k in range(num_class):
        sk_h = tf.slice(xs, [batch_size * k, 0], [batch_size, -1])
        _, var_tensor = tf.nn.moments(sk_h, axes=0)
        tmp_loss = tf.reduce_sum(var_tensor)
        total_loss = total_loss + batch_size * tmp_loss
    return total_loss * center_param

def outer_intra_loss(num_class, fc_feature, batch_size):
    ret_loss = 0.0
    intra_loss = 0.0
    s0_h = tf.slice(fc_feature, [0, 0], [2*batch_size, -1])
    s0_mean = tf.reduce_mean(s0_h, axis=0, keepdims=True)
    for k in range(1, num_class):
        exec("s%s_h = tf.slice(fc_feature, [batch_size*%s*2, 0], [batch_size*2, -1])"%(k, k))
        exec("s%s_mean = tf.reduce_mean(s%s_h, axis=0, keepdims=True)"%(k, k))
        exec("intra_loss = intra_loss + tf.reduce_mean(tf.reduce_sum(tf.pow(s%s_h - s%s_mean, 2), axis=1), axis=0)"%(k, k))
    total_mean = s0_mean
    intra_loss = intra_loss / num_class
    for k in range(1, num_class):
        exec("total_mean = tf.concat([total_mean, s%s_mean], 0)"%(k))
    for k in range(num_class):
        exec("res%s_mean = total_mean - s%s_mean"%(k, k))
        exec("ret_loss = ret_loss + tf.reduce_sum(tf.pow(res%s_mean, 2)) / 9.0"%(k))
    ret_loss = ret_loss / num_class
    return ret_loss / intra_loss
    #return intra_loss / ret_loss

"""
def outer_loss(num_class, fc_feature, batch_size):
    ret_loss = 0.0
    s0_h = tf.slice(fc_feature, [0, 0], [2*batch_size, -1])
    s0_mean = tf.reduce_mean(s0_h, axis=0, keepdims=True)
    for k in range(1, num_class):
        exec("s%s_h = tf.slice(fc_feature, [batch_size*%s*2, 0], [batch_size*2, -1])"%(k, k))
        exec("s%s_mean = tf.reduce_mean(s%s_h, axis=0, keepdims=True)"%(k, k))
    total_mean = s0_mean
    for k in range(1, num_class):
        exec("total_mean = tf.concat([total_mean, s%s_mean], 0)"%(k))
    for k in range(num_class):
        exec("res%s_mean = total_mean - s%s_mean"%(k, k))
        exec("ret_loss = ret_loss + tf.reduce_sum(tf.pow(res%s_mean, 2)) / 9.0"%(k))
    ret_loss = ret_loss / num_class
    return ret_loss
"""
def outer_loss(num_class, fc_feature, batch_size):
    ret_loss = 0.0
    s0_h = tf.slice(fc_feature, [0, 0], [2*batch_size, -1])
    s0_mean = tf.reduce_mean(s0_h, axis=0, keepdims=True)
    for k in range(1, num_class):
        exec("s%s_h = tf.slice(fc_feature, [batch_size*%s*2, 0], [batch_size*2, -1])"%(k, k))
        exec("s%s_mean = tf.reduce_mean(s%s_h, axis=0, keepdims=True)"%(k, k))
    total_mean = s0_mean
    for k in range(1, num_class):
        exec("total_mean = tf.concat([total_mean, s%s_mean], 0)"%(k))
    for k in range(num_class):
        exec("res%s_mean = tf.pow(total_mean - s%s_mean, 2)"%(k, k))
        exec("ret_loss = ret_loss + tf.reduce_max(tf.reduce_sum(res%s_mean, axis=1))"%(k))
    ret_loss = ret_loss / num_class
    return ret_loss


def intra_outer_loss(num_class, fc_feature, batch_size):
    ret_loss = 0.0
    intra_loss = 0.0
    s0_h = tf.slice(fc_feature, [0, 0], [2*batch_size, -1])
    s0_mean = tf.reduce_mean(s0_h, axis=0, keepdims=True)
    for k in range(1, num_class):
        exec("s%s_h = tf.slice(fc_feature, [batch_size*%s*2, 0], [batch_size*2, -1])"%(k, k))
        exec("s%s_mean = tf.reduce_mean(s%s_h, axis=0, keepdims=True)"%(k, k))
        exec("intra_loss = intra_loss + tf.reduce_mean(tf.reduce_sum(tf.pow(s%s_h - s%s_mean, 2), axis=1), axis=0)"%(k, k))
    total_mean = s0_mean
    intra_loss = intra_loss / num_class
    for k in range(1, num_class):
        exec("total_mean = tf.concat([total_mean, s%s_mean], 0)"%(k))
    for k in range(num_class):
        exec("res%s_mean = total_mean - s%s_mean"%(k, k))
        exec("ret_loss = ret_loss + tf.reduce_sum(tf.pow(res%s_mean, 2)) / 9.0"%(k))
    ret_loss = ret_loss / num_class
    #return ret_loss / intra_loss
    return intra_loss / ret_loss

def closs_outer(num_class, fc_feature, batch_size):
    ret_loss = 0.0
    s0_h  = tf.slice(fc_feature, [0, 0], [batch_size*2, -1])
    _D_0 = tf.reduce_sum(s0_h, axis=0, keep_dims=True)
    C_0 = (tf.matmul(tf.transpose(s0_h), s0_h) - tf.matmul(tf.transpose(_D_0), _D_0) / (batch_size)) / (batch_size - 1)
    C_0 = tf.expand_dims(C_0, 0)
    C_total = C_0
    for k in range(1, num_class):
        exec("s%s_h = tf.slice(fc_feature, [batch_size*%s*2, 0], [batch_size*2, -1])"%(k, k))
        exec("_D_%s = tf.reduce_sum(s%s_h, axis=0, keep_dims=True)"%(k, k))
        exec("C_%s = (tf.matmul(tf.transpose(s%s_h), s%s_h) - tf.matmul(tf.transpose(_D_%s), _D_%s) / (batch_size))/"
             "(batch_size-1)"%(k, k, k, k, k))
        exec("C_%s = tf.expand_dims(C_%s, 0)"%(k, k))
        exec("C_total = tf.concat([C_total, C_%s], 0)"%(k))
    for k in range(num_class):
        exec("tmp_loss = tf.nn.l2_loss(C_total - C_%s)"%(k))
        exec("ret_loss = ret_loss + tmp_loss")
    return ret_loss/20.0


def foo(data, p):
    import random
    import numpy as np

    assert 0 <= p <= 1

    data = data.flatten()
    assert data.size == 784

    n_dim = data.size
    random_inds = random.sample(range(n_dim), int(n_dim * p))
    data[random_inds] = np.random.rand(len(random_inds))
    return data


def foo_mnist(data, p=0.4):
    num_sample = len(data)
    
    for i in range(num_sample):
        data[i] = foo(data[i], p)

    return data