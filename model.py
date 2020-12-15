import os
import numpy as np
import tensorflow as tf
import tf_slim as slim
from tensorflow.compat.v1.nn.rnn_cell import GRUCell


class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        self.with_loss = False
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.neg_num = 10
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.compat.v1.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.mask = tf.compat.v1.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.compat.v1.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.compat.v1.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.compat.v1.get_variable("mid_embedding_var", [n_mid, embedding_dim], trainable=True)
            self.mid_embeddings_bias = tf.compat.v1.get_variable("bias_lookup_table", [n_mid], initializer=tf.zeros_initializer(), trainable=False)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))


    def build_sampled_softmax_loss(self, item_emb, user_emb, attn_loss=None):
        loss_vect = tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid)
        if self.with_loss:
            self.loss = tf.reduce_mean(loss_vect + attn_loss)
        else: 
            self.loss = tf.reduce_mean(loss_vect)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.lr: inps[4]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.mask: inps[1]
        })
        return user_embs
    
    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, path + 'model.ckpt')                  

    def restore(self, sess, path):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)

class Model_DNN(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_DNN, self).__init__(n_mid, embedding_dim, hidden_size,
                                           batch_size, seq_len, flag="DNN")

        masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1)

        self.item_his_eb_mean = tf.reduce_sum(self.item_his_eb, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        self.user_eb = tf.compat.v1.layers.dense(self.item_his_eb_mean, hidden_size, activation=None)
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)

class Model_GRU4REC(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_GRU4REC, self).__init__(n_mid, embedding_dim, hidden_size,
                                           batch_size, seq_len, flag="GRU4REC")
        with tf.name_scope('rnn_1'):
            self.sequence_length = self.mask_length
            rnn_outputs, final_state1 = tf.compat.v1.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")

        self.user_eb = final_state1
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

class CapsuleNetwork(tf.compat.v1.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, item_eb, mask):
        with tf.compat.v1.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.compat.v1.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.compat.v1.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None, bias_initializer=None)
            else:
                w = tf.compat.v1.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(tf.compat.v1.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.compat.v1.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout

class Model_MIND(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, hard_readout=True, relu_layer=True):
        super(Model_MIND, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="MIND")

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=0, num_interest=num_interest, hard_readout=hard_readout, relu_layer=relu_layer)
        self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)

        self.build_sampled_softmax_loss(self.item_eb, self.readout)

class Model_ComiRec_DR(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, hard_readout=True, relu_layer=False):
        super(Model_ComiRec_DR, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="ComiRec_DR")

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=2, num_interest=num_interest, hard_readout=hard_readout, relu_layer=relu_layer)
        self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)

        self.build_sampled_softmax_loss(self.item_eb, self.readout)

class Model_ComiRec_SA(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True, with_loss=False):
        super(Model_ComiRec_SA, self).__init__(n_mid, embedding_dim, hidden_size,
                                                   batch_size, seq_len, flag="ComiRec_SA")
        self.with_loss = with_loss
        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim])

        if add_pos:
            self.position_embedding = \
                tf.compat.v1.get_variable(
                    shape=[1, seq_len, embedding_dim],
                    name='position_embedding')
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = num_interest
        with tf.compat.v1.variable_scope("self_atten", reuse=tf.compat.v1.AUTO_REUSE) as scope:
            item_hidden = tf.compat.v1.layers.dense(item_list_add_pos, hidden_size * 4, activation=tf.nn.tanh)
            item_att_w  = tf.compat.v1.layers.dense(item_hidden, num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            item_att_w = tf.nn.softmax(item_att_w)
            self.seq_attn_weight = item_att_w
            interest_emb = tf.matmul(item_att_w, item_list_emb)

        self.user_eb = interest_emb

        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))

        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_list_emb)[0]) * num_heads)

        if self.with_loss:
            user_attn = self.diff_output(self.user_eb)
            self.build_sampled_softmax_loss(self.item_eb, readout, self.user_eb)
        else:
            self.build_sampled_softmax_loss(self.item_eb, readout)

    def diff_output(self, user_emb):
        # [batch, channels, heads]
        with tf.name_scope('diff_outputs'):
            x = user_emb
            x = tf.transpose(x, [0, 2, 1])  #shape [batch, heads, channels]
            x = tf.nn.l2_normalize(x, axis=-1) #normalize the last dimension
            x1 = tf.expand_dims(x, 1)  #shape [batch, 1, heads, channels]
            x2 = tf.expand_dims(x, 2)  #shape [batch, heads, 1, channels]
            cos_diff = tf.reduce_sum(tf.multiply(x1, x2), axis=[-1]) #shape [batch, heads, heads], broadcasting
            cos_diff = tf.reduce_max(cos_diff, axis=[-2,-1]) # shape [batch]
            return cos_diff

