#!/usr/bin/env python
#encoding: utf-8
import sys
import colored
import imp
imp.reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("../")
from model_trainer.base_task import RNN, Attention_RNN1, Attention_RNN2, Attention_RNN3, Attention_RNN4, \
    Attention_RNN5, Attention_RNN6, CNN
from record import do_record
from confusion_matrix import Alphabet
from confusion_matrix import ConfusionMatrix
import datetime
from tensorflow.contrib import learn
import config
import data_helpers
import util
import tensorflow as tf
import numpy as np
import argparse
# from scorer import get_rank_score_by_file
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# from sys import version_infovim
# print(tf.__version__)
# print(version_info)
timestamp = time.time()


FLAGS=tf.app.flags.FLAGS
# Data set
# tf.flags.DEFINE_string("level1_sense", "Comparison", "level1_sense (default: Comparison)")
# tf.flags.DEFINE_string("dataset_type", "PDTB_imp", "dataset_type (default: PDTB_imp)")
tf.app.flags.DEFINE_string("train_data_dir", "", "train data dir")

# models
tf.app.flags.DEFINE_string("model", "RNN", "model(default: 'RNN')")

# Model Hyperparameters
'''  RNN '''
tf.app.flags.DEFINE_boolean("share_rep_weights", True, "share_rep_weights")
tf.app.flags.DEFINE_boolean("bidirectional", False, "bidirectional")
# cell
tf.app.flags.DEFINE_string("cell_type", "BasicLSTM", "Cell Type(default: 'BasicLSTM')")
tf.app.flags.DEFINE_integer("hidden_size", 50, "Number of Hidden Size (default: 100)")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of RNN Layer (default: 1)")

# Training parameters
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.app.flags.DEFINE_float("learning_rate", 0.005, "Learning Rate (default: 0.005)")

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 10)")

tf.app.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# embedding
tf.app.flags.DEFINE_string("embedding", "Glove", "embedding(default:'Glove')")


# FLAGS._parse_flags()
# FLAGS.flag_values_dict()
# FLAGS(sys.argv)
print("\nParameters:")
for attr, value in FLAGS.__flags.items():
    print(("{}={}".format(attr, value.value)))
print("")
bidirectional=False
print(FLAGS.train_data_dir)
print('model:', FLAGS.model)
print(bidirectional)
print(FLAGS.hidden_size)


model_mapping = {
    "RNN": RNN,
    "Attention_RNN1": Attention_RNN1,
    "Attention_RNN2": Attention_RNN2,
    "Attention_RNN3": Attention_RNN3,
    "Attention_RNN4": Attention_RNN4,
    "Attention_RNN5": Attention_RNN5,
    "Attention_RNN6": Attention_RNN6
}

# for recording

train_data_dir = FLAGS.train_data_dir
# if "conll" in train_data_dir:
#     if FLAGS.blind:
#         record_file = config.RECORD_PATH + "/base_task/conll_blind.csv"
#     else:
#         record_file = config.RECORD_PATH + "/base_task/conll.csv"
# elif "ZH" in train_data_dir:
#     if FLAGS.blind:
#         record_file = config.RECORD_PATH + "/base_task/zh_blind.csv"
#     else:
#         record_file = config.RECORD_PATH + "/base_task/zh.csv"
# else:
record_file = config.RECORD_PATH + "/base_task/four_way.csv"

print("==> record path: %s" % record_file)
print()

evaluation_result = {
    "f1": 0.0,
    "p": 0.0,
    "r": 0.0,
    "acc": 0.0
}
configuration = {
    "train_data_dir": FLAGS.train_data_dir,
    "embedding": FLAGS.embedding,
    "model": FLAGS.model,
    "share_rep_weights": FLAGS.share_rep_weights,
    "bidirectional": bidirectional,

    "cell_type": FLAGS.cell_type,
    "hidden_size": FLAGS.hidden_size,
    "num_layers": FLAGS.num_layers,

    "dropout_keep_prob": FLAGS.dropout_keep_prob,
    "l2_reg_lambda": FLAGS.l2_reg_lambda,
    "Optimizer": "AdaOptimizer",
    "learning_rate": FLAGS.learning_rate,

    "batch_size": FLAGS.batch_size,
    "num_epochs": FLAGS.num_epochs,

    "w2v_type":FLAGS.embedding,
}
additional_conf = {"BLLIP"}



# Data Preparation
# ==================================================

# Load data
print("Loading data...")
# level1_sense = FLAGS.level1_sense
# dataset_type = FLAGS.dataset_type



train_data_dir = FLAGS.train_data_dir   # 'D:/PY/Pycode/project/end-to-end-discourse-parser/data/four_way/PDTB_imp'
# 准备原始数据文本
train_arg1s, train_arg2s, train_labels = data_helpers.load_data_and_labels("%s/train" % train_data_dir)
dev_arg1s, dev_arg2s, dev_labels = data_helpers.load_data_and_labels("%s/dev" % train_data_dir)
test_arg1s, test_arg2s, test_labels = data_helpers.load_data_and_labels("%s/test" % train_data_dir)

num_classes = train_labels.shape[1]


print("num_classes", num_classes)

# Build vocabulary
max_document_length = 100
all_text = train_arg1s + train_arg2s + dev_arg1s + dev_arg2s + test_arg1s + test_arg2s
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(all_text)

# transform 将arg从list文本变为对应的词的编号的二维数组
# arg:
# 句子1：word1 word2 word3 ... wordn
# 句子2：。。。
# 句子n:...
# -------->[n,100]:
# [
#     [345,56,453...34,0,0,0,0,0](总共100维)
#     ...
#     [23,.....] (第n行)
# ]
train_arg1s = np.array(list(vocab_processor.transform(train_arg1s)))
train_arg2s = np.array(list(vocab_processor.transform(train_arg2s)))
dev_arg1s = np.array(list(vocab_processor.transform(dev_arg1s)))
dev_arg2s = np.array(list(vocab_processor.transform(dev_arg2s)))
test_arg1s = np.array(list(vocab_processor.transform(test_arg1s)))
test_arg2s = np.array(list(vocab_processor.transform(test_arg2s)))


# load word embedding matrix 词向量:(n,m)n为所有文本单词个数，即下面的Vocabulary Size，m为词向量维度，google_news中为300。
vocab_embeddings = \
    util.load_embedding(train_data_dir, vocab_processor.vocabulary_._mapping, FLAGS.embedding, from_origin=True)



print(("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_))))
print(("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(train_labels), len(dev_labels), len(test_labels))))


''' Training '''
# ==================================================

with tf.Graph().as_default():
    tf.set_random_seed(1)

    # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options
    )

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        if FLAGS.model == "CNN":
            model = CNN(
                w2v_length=train_arg1s.shape[1],
                vocab_embeddings=vocab_embeddings,
                num_classes=train_labels.shape[1],

                filter_sizes=[1, 3, 5],
                num_filters=100,
            )
        else:
            model = model_mapping[FLAGS.model](
                sent_length=train_arg1s.shape[1],
                vocab_embeddings=vocab_embeddings,
                num_classes=train_labels.shape[1],

                cell_type=FLAGS.cell_type,
                hidden_size=FLAGS.hidden_size,
                num_layers=FLAGS.num_layers,
                bidirectional=bidirectional,
                share_rep_weights=FLAGS.share_rep_weights,
                batch_size=FLAGS.batch_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                additional_conf = additional_conf
            )


        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        # write to logs
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        # tensorboard - -logdir = logs


        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # save model
        saver = tf.train.Saver(max_to_keep=1)

        def train_step(s1_batch, s2_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                model.input_s1: s1_batch,
                model.input_s2: s2_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, model.loss, model.accuracy],
                feed_dict)

            # np.set_printoptions(threshold=np.nan)
            # print "==" * 40
            # print outputs1[0].shape
            # print output_1[0].shape
            # print outputs1[0][:50]
            # print output_1[0]


            time_str = datetime.datetime.now().isoformat()
            print(("\r {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)),end='')


        def test_step(s1_all, s2_all, y_all, test_str):
            """
            Evaluates model on a dev/test set
            """
            golds = []
            predictions = []


            feed_dict = {
                model.input_s1: s1_all,
                model.input_s2: s2_all,
                model.input_y: y_all,
                model.dropout_keep_prob: 1.0
            }

            step, loss, accuracy, softmax_scores, curr_predictions, curr_golds = sess.run(
                [global_step, model.loss, model.accuracy, model.softmax_scores,
                 model.predictions, model.golds], feed_dict)

            # print('\t %s loss:' % test_str, loss)

            golds += list(curr_golds)  # 真实label:[2,1,2,2。。。]
            predictions += list(curr_predictions)


            alphabet = Alphabet()
            for i in range(num_classes):
                alphabet.add(str(i))
            confusionMatrix = ConfusionMatrix(alphabet)
            predictions = list(map(str, predictions))   #[2,3,1,0,2...2]-->['2','3','1','0','2'...'2']
            golds = list(map(str, golds))
            confusionMatrix.add_list(predictions, golds) # 将预测predictions和golds填入confusionMatrix.matrix的4*4表格
            confusionMatrix.loss = loss

            return confusionMatrix # 主要就是4*4表格


        def _prediction_on_dev(best_score, best_output_string):

            confusionMatrix = test_step(dev_arg1s, dev_arg2s, dev_labels, test_str='dev') #得到4*4的矩阵

            acc = confusionMatrix.get_accuracy()
            # 对着 f1 调
            p, r, f1 = confusionMatrix.get_average_prf()

            # current performance
            curr_output_string = confusionMatrix.get_matrix() + confusionMatrix.get_summary()
                                 # + confusionMatrix.get_micro_f1()

            flag = 0
            if f1 >= best_score:
                flag = 1
                best_score = f1

                best_output_string = confusionMatrix.get_matrix() + confusionMatrix.get_summary()
                                 # + confusionMatrix.get_micro_f1()

                # print("")
                # print("\nEvaluation on Test:")
                # confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels)
                # confusionMatrix.print_out()
                # print("")

                acc = confusionMatrix.get_accuracy()
                p, r, f1 = confusionMatrix.get_average_prf()

                evaluation_result["acc"] = "%.4f" % acc
                evaluation_result["f1"] = "%.4f" % f1
                evaluation_result["p"] = "%.4f" % p
                evaluation_result["r"] = "%.4f" % r

                # save model
                saver.save(sess, 'ckpt/best.ckpt')

            # color = colored.bg('black') + colored.fg('green')
            # reset = colored.attr('reset')

            print("\nEvaluation on Dev:")
            print("Current Performance:")
            print('\033[34m',curr_output_string, '\033[0m') #当前结果蓝色（34）显示

            if flag == 1:
                print("  " * 40 + '❤️')

            # print((color + 'Best Performance' + reset))
            # print((color + best_output_string + reset))
            print('\033[32m Best Performance\033[0m') #最佳结果绿色（32）显示
            print(('\033[32m' + best_output_string + '\033[0m'))

            return best_score, best_output_string


        def _prediction_on_test():
            # 恢复模型
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)

            confusionMatrix = test_step(dev_arg1s, dev_arg2s, dev_labels, test_str='dev') #得到4*4的矩阵
            acc = confusionMatrix.get_accuracy()
            curr_output_string = confusionMatrix.get_matrix() + confusionMatrix.get_summary()
            print("\nEvaluation on Dev:")
            print("Current Performance:")
            print('\033[33m',curr_output_string, '\033[0m') #当前结果黄色（33）显示



            confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels, test_str='test') #得到4*4的矩阵
            acc = confusionMatrix.get_accuracy()
            curr_output_string = confusionMatrix.get_matrix() + confusionMatrix.get_summary()
            print("\nEvaluation on test:")
            print("Current Performance:")
            print('\033[33m',curr_output_string, '\033[0m') #当前结果黄色（33）显示


        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(train_arg1s, train_arg2s, train_labels)), FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)

        best_score = 0.0
        best_output_string = ""
        # Training loop. For each batch...
        for batch in batches:
            s1_batch, s2_batch, y_batch = list(zip(*batch))
            train_step(s1_batch, s2_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                if num_classes == 4:
                    best_score, best_output_string = _prediction_on_dev(best_score, best_output_string)

        _prediction_on_test()



# record the configuration and result
fieldnames = ["f1", "p", "r", "acc", "train_data_dir", "embedding", "model", "share_rep_weights",
                   "bidirectional", "cell_type",  "hidden_size", "num_layers",
                  "dropout_keep_prob", "l2_reg_lambda", "Optimizer", "learning_rate", "batch_size", "num_epochs", "w2v_type",
                  "additional_conf"
]
do_record(fieldnames, configuration, additional_conf, evaluation_result, record_file)


