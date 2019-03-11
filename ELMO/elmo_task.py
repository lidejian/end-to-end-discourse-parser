#!/usr/bin/env python
#encoding: utf-8
import sys
import imp
imp.reload(sys)
# sys.setdefaultencoding('utf-8')
from . import util
import tensorflow as tf
import numpy as np

def _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob):
    cell = None
    if cell_type == "BasicLSTM":
        # cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    if cell_type == "LSTM":
        # cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0)
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
    if cell_type == "GRU":
        # cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        cell = tf.contrib.rnn.GRUCell(hidden_size)

    if cell is None:
        raise ValueError("cell type: %s is incorrect!!" % (cell_type))

    # dropout
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    # multi-layer
    if num_layers > 1:
        # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
    return cell


def _get_rnn(inputs,
            cell_type,
            hidden_size,
            num_layers,
            dropout_keep_prob,
            bidirectional=False):
    if bidirectional:
        cell_fw = _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob)
        cell_bw = _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=tf.to_int64(util.length(inputs)),
            dtype=tf.float32)

        hidden_size *= 2
        # outputs = tf.concat(2, outputs)     # 0.x版本：数字在前，tensors在后：tf.concat(n, tensors)
        outputs = tf.concat(outputs, 2)     # 1.0及以后版本：tensors在前，数字在后：tf.concat(tensors, n)


    else:
        cell = _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob)
        outputs, state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            dtype=tf.float32,
            sequence_length=util.length(inputs),
        )

    return outputs, state, hidden_size


# input_a: (None, d)
# w_shape: (d, d, k)
# input_b: (None, d)
def bilinear_production(input_a, w_shape, input_b):
    d, d, k = w_shape

    W = tf.get_variable(
        "bilinear_W",
        shape=[d, d, k],
        initializer=tf.contrib.layers.xavier_initializer())

    # (None, d) * (d, d*k) ==> (None, d*k) ==> (None, d, k)
    temp = tf.reshape(tf.matmul(input_a, tf.reshape(W, [d, d * k])), [-1, d, k])
    # (None, d, 1)
    input_b_temp = tf.expand_dims(input_b, 2)
    # (None, k)
    result = tf.reduce_sum(temp * input_b_temp, axis=1)

    return result

def CNN_Baseline():
    pass


class RNN(object):
    def __init__(self,
                 sent_length,
                 emb_size,
                 num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,

                 batch_size,

                 l2_reg_lambda=0,
                 additional_conf={},
                 ):

        self.input_s1 = tf.placeholder(tf.float32, [None, sent_length, emb_size], name="input_s1")
        self.input_s2 = tf.placeholder(tf.float32, [None, sent_length, emb_size], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # # Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)
        #     self.embedded_s1 = tf.nn.embedding_lookup(embedding, self.input_s1)
        #     self.embedded_s2 = tf.nn.embedding_lookup(embedding, self.input_s2)

        if share_rep_weights:
            with tf.variable_scope("RNN"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.input_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

                # share weights
                tf.get_variable_scope().reuse_variables()

                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.input_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )
        else:
            with tf.variable_scope("RNN1"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.input_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

            with tf.variable_scope("RNN2"):
                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.input_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

        # outputs1: (batch_size, num_steps, hidden_size)
        with tf.name_scope("output"):
            # simple concatenation + simple subtraction

            self.outputs1 = outputs1
            self.outputs2 = outputs2
            self.output_1 = util.last_relevant(outputs1, util.length(self.input_s1))
            self.output_2 = util.last_relevant(outputs2, util.length(self.input_s2))

            # self.output = tf.concat(1, [self.output_1, self.output_2])
            self.output = tf.concat([self.output_1, self.output_2], 1)   # 1.0及以后版本：tensors在前，数字在后
            self.size = hidden_size * 2

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("softmax"):
            W = tf.get_variable(
                "W",
                shape=[self.size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.golds = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class Attention_RNN1(object):
    def __init__(self,
                 sent_length,
                 vocab_embeddings,
                 num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,
                 batch_size,
                 l2_reg_lambda=0,
                 additional_conf={},
                 ):

        self.input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="input_s1")
        self.input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):

            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)
            self.embedded_s1 = tf.nn.embedding_lookup(embedding, self.input_s1)
            self.embedded_s2 = tf.nn.embedding_lookup(embedding, self.input_s2)

        if share_rep_weights:
            with tf.variable_scope("RNN"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

                # share weights
                tf.get_variable_scope().reuse_variables()

                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )
        else:
            with tf.variable_scope("RNN1"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

            with tf.variable_scope("RNN2"):
                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

        with tf.variable_scope("Attention"):

            W = tf.get_variable(
                "W",
                shape=[hidden_size, hidden_size],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            # Attention Pooling Network
            atten_left = tf.reshape(tf.matmul(tf.reshape(outputs1, [-1, hidden_size]), W),
                                    [batch_size, sent_length, hidden_size])
            atten_right = tf.nn.tanh(tf.matmul(atten_left, tf.transpose(outputs2, perm=[0,2,1])))

            max_pooled_s1 = tf.nn.softmax(tf.reduce_max(atten_right, reduction_indices=[2]))
            max_pooled_s2 = tf.nn.softmax(tf.reduce_max(atten_right, reduction_indices=[1]))

            attention_s1 = tf.matmul(tf.transpose(outputs1,perm=[0,2,1]), tf.reshape(max_pooled_s1, [batch_size, sent_length, 1]))
            attention_s2 = tf.matmul(tf.transpose(outputs2,perm=[0,2,1]), tf.reshape(max_pooled_s2, [batch_size, sent_length, 1]))

        # self.output = tf.reshape(tf.concat(1, [attention_s1, attention_s2]), [batch_size, 2 * hidden_size])
        self.output = tf.reshape(tf.concat([attention_s1, attention_s2],1), [batch_size, 2 * hidden_size])# 1.0及以后版本：tensors在前，数字在后

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("softmax"):
            W = tf.get_variable(
                "W",
                shape=[2 * hidden_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.golds = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# my simple attention neural network
class Attention_RNN2(object):
    def __init__(self,
                 sent_length,
                 vocab_embeddings,
                 num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,

                 batch_size,

                 l2_reg_lambda=0,
                 additional_conf={},
                 ):

        self.input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="input_s1")
        self.input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)
            self.embedded_s1 = tf.nn.embedding_lookup(embedding, self.input_s1)
            self.embedded_s2 = tf.nn.embedding_lookup(embedding, self.input_s2)

        if share_rep_weights:
            with tf.variable_scope("RNN"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

                # share weights
                tf.get_variable_scope().reuse_variables()

                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )
        else:
            with tf.variable_scope("RNN1"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

            with tf.variable_scope("RNN2"):
                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

        # outputs1: (batch_size, num_steps, hidden_size)
        with tf.name_scope("attention"):
            # simple concatenation + simple subtraction

            # (None, sent_length, hidden_size)
            self.outputs1 = outputs1
            self.outputs2 = outputs2

            # # 1. 取 rnn 的最后一个hidden state
            # # (None, hidden_size)
            # self.r1 = util.last_relevant(outputs1, util.length(self.embedded_s1))
            # self.r2 = util.last_relevant(outputs2, util.length(self.embedded_s2))

            # 2. sum 所有的hidden state
            self.r1 = tf.reduce_sum(self.outputs1, 1)
            self.r2 = tf.reduce_sum(self.outputs2, 1)


            # (None, 1, hidden_size)
            r2_temp = tf.transpose(tf.expand_dims(self.r2, -1), [0, 2, 1])
            # (None, sent_length)
            p1 = tf.nn.softmax(tf.reduce_sum(self.outputs1 * r2_temp, 2))

            # (None, 1, hidden_size)
            r1_temp = tf.transpose(tf.expand_dims(self.r1, -1), [0, 2, 1])
            # (None, sent_length)
            p2 = tf.nn.softmax(tf.reduce_sum(self.outputs2 * r1_temp, 2))


            # (None, 1, sent_length)
            p1_temp =  tf.transpose(tf.expand_dims(p1, -1), [0, 2, 1])
            # (None, hidden_size, sent_length)
            outputs1_temp = tf.transpose(outputs1, [0, 2, 1])
            # (None, hidden_size)
            self.v1 = tf.reduce_sum(outputs1_temp * p1_temp, 2)

            # (None, 1, sent_length)
            p2_temp = tf.transpose(tf.expand_dims(p2, -1), [0, 2, 1])
            # (None, hidden_size, sent_length)
            outputs2_temp = tf.transpose(outputs2, [0, 2, 1])
            # (None, hidden_size)
            self.v2 = tf.reduce_sum(outputs2_temp * p2_temp, 2)


            self.output = tf.concat([self.v1, self.v2],1)  # 1.0及以后版本：tensors在前，数字在后
            self.size = hidden_size * 2

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("softmax"):
            W = tf.get_variable(
                "W",
                shape=[self.size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            tf.summary.scalar('loss', self.loss)

            # Accuracy
        with tf.name_scope("accuracy"):
            self.golds = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            tf.summary.scalar('accuracy', self.accuracy)



# my simple attention neural network, multi-layer attention
class Attention_RNN3(object):
    def __init__(self,
                 sent_length,
                 vocab_embeddings,
                 num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,

                 batch_size,

                 l2_reg_lambda=0,
                 additional_conf={},
                 ):

        self.input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="input_s1")
        self.input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)
            self.embedded_s1 = tf.nn.embedding_lookup(embedding, self.input_s1)
            self.embedded_s2 = tf.nn.embedding_lookup(embedding, self.input_s2)

        if share_rep_weights:
            with tf.variable_scope("RNN"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

                # share weights
                tf.get_variable_scope().reuse_variables()

                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )
        else:
            with tf.variable_scope("RNN1"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

            with tf.variable_scope("RNN2"):
                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

        # outputs1: (batch_size, num_steps, hidden_size)
        with tf.name_scope("attention"), tf.variable_scope("attention"):

            s1_attention, s2_attention = self._attention(outputs1, outputs2)

            # RNN on s1_attention and s2_attention
            if share_rep_weights:
                with tf.variable_scope("RNN"):
                    attention_outputs1, states1, _ = _get_rnn(
                        inputs=s1_attention,
                        cell_type=cell_type,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout_keep_prob=self.dropout_keep_prob,
                        bidirectional=bidirectional
                    )

                    # share weights
                    tf.get_variable_scope().reuse_variables()

                    attention_outputs2, states2, hidden_size = _get_rnn(
                        inputs=s2_attention,
                        cell_type=cell_type,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout_keep_prob=self.dropout_keep_prob,
                        bidirectional=bidirectional
                    )
            else:
                with tf.variable_scope("RNN1"):
                    attention_outputs1, states1, _ = _get_rnn(
                        inputs=s1_attention,
                        cell_type=cell_type,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout_keep_prob=self.dropout_keep_prob,
                        bidirectional=bidirectional
                    )

                with tf.variable_scope("RNN2"):
                    attention_outputs2, states2, hidden_size = _get_rnn(
                        inputs=s2_attention,
                        cell_type=cell_type,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout_keep_prob=self.dropout_keep_prob,
                        bidirectional=bidirectional
                    )

            # (None, sent_length, hidden_size)
            attention_outputs1, attention_outputs2 = self._attention(attention_outputs1, attention_outputs2)

            # (None, hidden_size)
            self.v1 = tf.reduce_sum(attention_outputs1, 1)
            self.v2 = tf.reduce_sum(attention_outputs2, 1)

            self.output = tf.concat([self.v1, self.v2],1)# 1.0及以后版本：tensors在前，数字在后
            self.size = hidden_size * 2

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("softmax"):
            W = tf.get_variable(
                "W",
                shape=[self.size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.golds = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



    def _attention(self, input_matrix_1, input_matrix_2):


        # input_matrix_1: (None, sent_length, hidden_size)
        # input_matrix_2: (None, sent_length, hidden_size)

        # # 1. 取 rnn 的最后一个hidden state
        # # (None, hidden_size)
        # self.r1 = util.last_relevant(outputs1, util.length(self.embedded_s1))
        # self.r2 = util.last_relevant(outputs2, util.length(self.embedded_s2))

        # 2. sum 所有的hidden state
        # (None, hidden_size)
        r1 = tf.reduce_sum(input_matrix_1, 1)
        r2 = tf.reduce_sum(input_matrix_2, 1)

        # (None, 1, hidden_size)
        r2_temp = tf.transpose(tf.expand_dims(r2, -1), [0, 2, 1])
        # (None, sent_length)
        p1 = tf.nn.softmax(tf.reduce_sum(input_matrix_1 * r2_temp, 2))

        # (None, 1, hidden_size)
        r1_temp = tf.transpose(tf.expand_dims(r1, -1), [0, 2, 1])
        # (None, sent_length)
        p2 = tf.nn.softmax(tf.reduce_sum(input_matrix_2 * r1_temp, 2))

        # (None, 1, sent_length)
        p1_temp = tf.transpose(tf.expand_dims(p1, -1), [0, 2, 1])
        # (None, hidden_size, sent_length)
        outputs1_temp = tf.transpose(input_matrix_1, [0, 2, 1])
        # (None, sent_length, hidden_size)
        output_matrix_1 = tf.transpose(outputs1_temp * p1_temp, [0, 2, 1])

        # (None, 1, sent_length)
        p2_temp = tf.transpose(tf.expand_dims(p2, -1), [0, 2, 1])
        # (None, hidden_size, sent_length)
        outputs2_temp = tf.transpose(input_matrix_2, [0, 2, 1])
        # (None, sent_length, hidden_size)
        output_matrix_2 = tf.transpose(outputs2_temp * p2_temp, [0, 2, 1])

        return output_matrix_1, output_matrix_2


# my simple attention neural network, bilinear
class Attention_RNN4(object):
    def __init__(self,
                 sent_length,
                 vocab_embeddings,
                 num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,

                 batch_size,

                 l2_reg_lambda=0,
                 additional_conf={},
                 ):

        self.input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="input_s1")
        self.input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)
            self.embedded_s1 = tf.nn.embedding_lookup(embedding, self.input_s1)
            self.embedded_s2 = tf.nn.embedding_lookup(embedding, self.input_s2)

        if share_rep_weights:
            with tf.variable_scope("RNN"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

                # share weights
                tf.get_variable_scope().reuse_variables()

                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )
        else:
            with tf.variable_scope("RNN1"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

            with tf.variable_scope("RNN2"):
                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

        # outputs1: (batch_size, num_steps, hidden_size)
        with tf.name_scope("attention"):
            # simple concatenation + simple subtraction

            # (None, sent_length, hidden_size)
            self.outputs1 = outputs1
            self.outputs2 = outputs2

            # # 1. 取 rnn 的最后一个hidden state
            # # (None, hidden_size)
            # self.r1 = util.last_relevant(outputs1, util.length(self.embedded_s1))
            # self.r2 = util.last_relevant(outputs2, util.length(self.embedded_s2))

            # 2. sum 所有的hidden state
            self.r1 = tf.reduce_sum(self.outputs1, 1)
            self.r2 = tf.reduce_sum(self.outputs2, 1)


            # (None, 1, hidden_size)
            r2_temp = tf.transpose(tf.expand_dims(self.r2, -1), [0, 2, 1])
            # (None, sent_length)
            p1 = tf.nn.softmax(tf.reduce_sum(self.outputs1 * r2_temp, 2))

            # (None, 1, hidden_size)
            r1_temp = tf.transpose(tf.expand_dims(self.r1, -1), [0, 2, 1])
            # (None, sent_length)
            p2 = tf.nn.softmax(tf.reduce_sum(self.outputs2 * r1_temp, 2))


            # (None, 1, sent_length)
            p1_temp =  tf.transpose(tf.expand_dims(p1, -1), [0, 2, 1])
            # (None, hidden_size, sent_length)
            outputs1_temp = tf.transpose(outputs1, [0, 2, 1])
            # (None, hidden_size)
            self.v1 = tf.reduce_sum(outputs1_temp * p1_temp, 2)

            # (None, 1, sent_length)
            p2_temp = tf.transpose(tf.expand_dims(p2, -1), [0, 2, 1])
            # (None, hidden_size, sent_length)
            outputs2_temp = tf.transpose(outputs2, [0, 2, 1])
            # (None, hidden_size)
            self.v2 = tf.reduce_sum(outputs2_temp * p2_temp, 2)

            k = 50
            # (None, k)
            self.v_bi = bilinear_production(self.v1, (hidden_size, hidden_size, k), self.v2)

            self.output = tf.concat([self.v1, self.v2, tf.subtract(self.v1, self.v2), self.v_bi],1)# 1.0及以后版本：tensors在前，数字在后
            self.size = hidden_size * 3 + k

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("softmax"):
            W = tf.get_variable(
                "W",
                shape=[self.size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.golds = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")




# my simple attention neural network
class Attention_RNN5(object):
    def __init__(self,
                 sent_length,
                 vocab_embeddings,
                 num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,

                 batch_size,

                 l2_reg_lambda=0,
                 additional_conf={},
                 ):

        self.input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="input_s1")
        self.input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)
            self.embedded_s1 = tf.nn.embedding_lookup(embedding, self.input_s1)
            self.embedded_s2 = tf.nn.embedding_lookup(embedding, self.input_s2)

        if share_rep_weights:
            with tf.variable_scope("RNN"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

                # share weights
                tf.get_variable_scope().reuse_variables()

                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )
        else:
            with tf.variable_scope("RNN1"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

            with tf.variable_scope("RNN2"):
                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

        # outputs1: (batch_size, num_steps, hidden_size)
        with tf.name_scope("attention"):
            # simple concatenation + simple subtraction

            # (None, sent_length, hidden_size)
            self.outputs1 = outputs1
            self.outputs2 = outputs2

            # 1. 取 rnn 的最后一个hidden state
            # (None, hidden_size)
            self.r1 = util.last_relevant(outputs1, util.length(self.embedded_s1))
            self.r2 = util.last_relevant(outputs2, util.length(self.embedded_s2))

            # # 2. sum 所有的hidden state
            # self.r1 = tf.reduce_sum(self.outputs1, 1)
            # self.r2 = tf.reduce_sum(self.outputs2, 1)

            W = tf.get_variable(
                "bilinear_attention_W",
                shape=[ hidden_size, hidden_size ],
                initializer=tf.contrib.layers.xavier_initializer())

            self.r1 = tf.matmul(self.r1, W)
            self.r2 = tf.matmul(self.r2, W)

            # (None, 1, hidden_size)
            r2_temp = tf.transpose(tf.expand_dims(self.r2, -1), [0, 2, 1])
            # (None, sent_length)
            p1 = tf.nn.softmax(tf.reduce_sum(self.outputs1 * r2_temp, 2))

            # (None, 1, hidden_size)
            r1_temp = tf.transpose(tf.expand_dims(self.r1, -1), [0, 2, 1])
            # (None, sent_length)
            p2 = tf.nn.softmax(tf.reduce_sum(self.outputs2 * r1_temp, 2))


            # (None, 1, sent_length)
            p1_temp =  tf.transpose(tf.expand_dims(p1, -1), [0, 2, 1])
            # (None, hidden_size, sent_length)
            outputs1_temp = tf.transpose(outputs1, [0, 2, 1])
            # (None, hidden_size)
            self.v1 = tf.reduce_sum(outputs1_temp * p1_temp, 2)

            # (None, 1, sent_length)
            p2_temp = tf.transpose(tf.expand_dims(p2, -1), [0, 2, 1])
            # (None, hidden_size, sent_length)
            outputs2_temp = tf.transpose(outputs2, [0, 2, 1])
            # (None, hidden_size)
            self.v2 = tf.reduce_sum(outputs2_temp * p2_temp, 2)


            self.output = tf.concat([self.v1, self.v2],1)# 1.0及以后版本：tensors在前，数字在后
            self.size = hidden_size * 2

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("softmax"):
            W = tf.get_variable(
                "W",
                shape=[self.size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.golds = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")




# my simple attention neural network + mlp
class Attention_RNN6(object):
    def __init__(self,
                 sent_length,
                 vocab_embeddings,
                 num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,

                 batch_size,

                 l2_reg_lambda=0,
                 additional_conf={},
                 ):

        self.input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="input_s1")
        self.input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)
            self.embedded_s1 = tf.nn.embedding_lookup(embedding, self.input_s1)
            self.embedded_s2 = tf.nn.embedding_lookup(embedding, self.input_s2)

        if share_rep_weights:
            with tf.variable_scope("RNN"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

                # share weights
                tf.get_variable_scope().reuse_variables()

                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )
        else:
            with tf.variable_scope("RNN1"):
                outputs1, states1, _ = _get_rnn(
                    inputs=self.embedded_s1,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

            with tf.variable_scope("RNN2"):
                outputs2, states2, hidden_size = _get_rnn(
                    inputs=self.embedded_s2,
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_keep_prob=self.dropout_keep_prob,
                    bidirectional=bidirectional
                )

        # outputs1: (batch_size, num_steps, hidden_size)
        with tf.name_scope("attention"), tf.variable_scope("attention"):
            # simple concatenation + simple subtraction

            # (None, sent_length, hidden_size)
            self.outputs1 = outputs1
            self.outputs2 = outputs2

            # # 1. 取 rnn 的最后一个hidden state
            # # (None, hidden_size)
            # self.r1 = util.last_relevant(outputs1, util.length(self.embedded_s1))
            # self.r2 = util.last_relevant(outputs2, util.length(self.embedded_s2))

            # 2. sum 所有的hidden state
            self.r1 = tf.reduce_sum(self.outputs1, 1)
            self.r2 = tf.reduce_sum(self.outputs2, 1)


            # (None, 1, hidden_size)
            r2_temp = tf.transpose(tf.expand_dims(self.r2, -1), [0, 2, 1])
            # (None, sent_length)
            p1 = tf.nn.softmax(tf.reduce_sum(self.outputs1 * r2_temp, 2))

            # (None, 1, hidden_size)
            r1_temp = tf.transpose(tf.expand_dims(self.r1, -1), [0, 2, 1])
            # (None, sent_length)
            p2 = tf.nn.softmax(tf.reduce_sum(self.outputs2 * r1_temp, 2))


            # (None, 1, sent_length)
            p1_temp =  tf.transpose(tf.expand_dims(p1, -1), [0, 2, 1])
            # (None, hidden_size, sent_length)
            outputs1_temp = tf.transpose(outputs1, [0, 2, 1])
            # (None, hidden_size)
            self.v1 = tf.reduce_sum(outputs1_temp * p1_temp, 2)

            # (None, 1, sent_length)
            p2_temp = tf.transpose(tf.expand_dims(p2, -1), [0, 2, 1])
            # (None, hidden_size, sent_length)
            outputs2_temp = tf.transpose(outputs2, [0, 2, 1])
            # (None, hidden_size)
            self.v2 = tf.reduce_sum(outputs2_temp * p2_temp, 2)


            v = tf.concat([self.v1, self.v2],1)# 1.0及以后版本：tensors在前，数字在后
            v_size = hidden_size * 2

            mlp_hidden_size = 50
            W = tf.get_variable(
                "W",
                shape=[v_size, mlp_hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[mlp_hidden_size]), name="b")

            self.output = tf.nn.tanh(tf.nn.xw_plus_b(v, W, b))
            self.size = mlp_hidden_size

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("softmax"):
            W = tf.get_variable(
                "W",
                shape=[self.size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.golds = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class CNN(object):

    """
        A CNN for text classification.
        Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self,
            sent_length,
            emb_size,
            num_classes,
            filter_sizes,
            num_filters,
            l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_s1 = tf.placeholder(tf.float32, [None, sent_length, emb_size], name="input_s1")
        self.input_s2 = tf.placeholder(tf.float32, [None, sent_length, emb_size], name="input_s2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)
        #     self.embedded_s1 = tf.nn.embedding_lookup(embedding, self.input_s1)
        #     self.embedded_s2 = tf.nn.embedding_lookup(embedding, self.input_s2)

        self.embedded_s1_expanded = tf.expand_dims(self.input_s1, -1)
        self.embedded_s2_expanded = tf.expand_dims(self.input_s2, -1)

        embedding_size = emb_size

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_s1 = []
        pooled_outputs_s2 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv_s1 = tf.nn.conv2d(
                    self.embedded_s1_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv1")

                conv_s2 = tf.nn.conv2d(
                    self.embedded_s2_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")

                # Apply nonlinearity
                h_s1 = tf.nn.tanh(tf.nn.bias_add(conv_s1, b), name="tanh_s1")
                h_s2 = tf.nn.tanh(tf.nn.bias_add(conv_s2, b), name="tanh_s2")


                pooled_s1 = tf.nn.max_pool(
                    h_s1,
                    ksize=[1, sent_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_s1")
                pooled_outputs_s1.append(pooled_s1)

                pooled_s2 = tf.nn.max_pool(
                    h_s2,
                    ksize=[1, sent_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_s2")
                pooled_outputs_s2.append(pooled_s2)

        # Combine all the pooled features
        num_filters_total = 2 * num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs_s1+pooled_outputs_s2,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.softmax_scores = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.golds = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
