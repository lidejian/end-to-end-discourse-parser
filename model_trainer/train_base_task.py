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
from scorer import get_rank_score_by_file
import time


# from sys import version_infovim
# print(tf.__version__)
# print(version_info)
timestamp = time.time()

# Data set
# tf.flags.DEFINE_string("level1_sense", "Comparison", "level1_sense (default: Comparison)")
# tf.flags.DEFINE_string("dataset_type", "PDTB_imp", "dataset_type (default: PDTB_imp)")
tf.flags.DEFINE_string("train_data_dir", "", "train data dir")
tf.flags.DEFINE_boolean("blind", False, "blind(default: 'False')")

# models
tf.flags.DEFINE_string("model", "RNN", "model(default: 'RNN')")

# Model Hyperparameters
'''  RNN '''
tf.flags.DEFINE_boolean("share_rep_weights", True, "share_rep_weights")
tf.flags.DEFINE_boolean("bidirectional", True, "bidirectional")
# cell
tf.flags.DEFINE_string("cell_type", "BasicLSTM", "Cell Type(default: 'BasicLSTM')")
tf.flags.DEFINE_integer("hidden_size", 100, "Number of Hidden Size (default: 100)")
tf.flags.DEFINE_integer("num_layers", 1, "Number of RNN Layer (default: 1)")

# Training parameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.005, "Learning Rate (default: 0.005)")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 10)")

tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
print("")



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
if "binary" in train_data_dir:
    record_file = config.RECORD_PATH + "/single_task/%s.csv" % train_data_dir.split("/")[-1]
elif "conll" in train_data_dir:
    if FLAGS.blind:
        record_file = config.RECORD_PATH + "/single_task/conll_blind.csv"
    else:
        record_file = config.RECORD_PATH + "/single_task/conll.csv"
elif "QA" in train_data_dir:
    record_file = config.RECORD_PATH + "/single_task/QA.csv"
elif "QQ" in train_data_dir:
    record_file = config.RECORD_PATH + "/single_task/QQ.csv"
elif "ZH" in train_data_dir:
    if FLAGS.blind:
        record_file = config.RECORD_PATH + "/single_task/zh_blind.csv"
    else:
        record_file = config.RECORD_PATH + "/single_task/zh.csv"
else:
    record_file = config.RECORD_PATH + "/single_task/four_way.csv"

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
    "model": FLAGS.model,
    "share_rep_weights": FLAGS.share_rep_weights,
    "bidirectional": FLAGS.bidirectional,

    "cell_type": FLAGS.cell_type,
    "hidden_size": FLAGS.hidden_size,
    "num_layers": FLAGS.num_layers,

    "dropout_keep_prob": FLAGS.dropout_keep_prob,
    "l2_reg_lambda": FLAGS.l2_reg_lambda,
    "Optimizer": "AdaOptimizer",
    "learning_rate": FLAGS.learning_rate,

    "batch_size": FLAGS.batch_size,
    "num_epochs": FLAGS.num_epochs,

    "w2v_type": "cqa 100维",
}
additional_conf = {}



# Data Preparation
# ==================================================

# Load data
print("Loading data...")
# level1_sense = FLAGS.level1_sense
# dataset_type = FLAGS.dataset_type

train_data_dir = FLAGS.train_data_dir

train_arg1s, train_arg2s, train_labels = data_helpers.load_data_and_labels("%s/train" % train_data_dir)
dev_arg1s, dev_arg2s, dev_labels = data_helpers.load_data_and_labels("%s/dev" % train_data_dir)
if FLAGS.blind:
    test_arg1s, test_arg2s, test_labels = data_helpers.load_data_and_labels("%s/blind_test" % train_data_dir)
else:
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
    util.load_google_word2vec_for_vocab(train_data_dir, vocab_processor.vocabulary_._mapping, from_origin=True)



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
                filter_sizes=[4, 6, 13],
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
                bidirectional=FLAGS.bidirectional,
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

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

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
            print(("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)))

        def test_step(s1_all, s2_all, y_all):
            """
            Evaluates model on a dev/test set
            """
            golds = []
            predictions = []

            n = len(s1_all)
            batch_size = FLAGS.batch_size
            start_index = 0
            while start_index < n:
                if start_index + batch_size <= n:
                    s1_batch = s1_all[start_index: start_index + batch_size]
                    s2_batch = s2_all[start_index: start_index + batch_size]
                    y_batch = y_all[start_index: start_index + batch_size]

                    feed_dict = {
                        model.input_s1: s1_batch,
                        model.input_s2: s2_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: 1.0
                    }

                    step, loss, accuracy, softmax_scores, curr_predictions, curr_golds = sess.run(
                        [global_step, model.loss, model.accuracy, model.softmax_scores,
                         model.predictions, model.golds], feed_dict)

                    golds += list(curr_golds)
                    predictions += list(curr_predictions)

                else:
                    left_num = n - start_index
                    # 填充一下
                    s1_batch = np.concatenate((s1_all[start_index:], s1_all[:batch_size - left_num]), axis=0)
                    s2_batch = np.concatenate((s2_all[start_index:], s2_all[:batch_size - left_num]), axis=0)
                    y_batch = np.concatenate((y_all[start_index:], y_all[:batch_size - left_num]), axis=0)

                    feed_dict = {
                        model.input_s1: s1_batch,
                        model.input_s2: s2_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy, softmax_scores, curr_predictions, curr_golds = sess.run(
                        [global_step, model.loss, model.accuracy, model.softmax_scores,
                         model.predictions, model.golds], feed_dict)

                    golds += list(curr_golds[:left_num])
                    predictions += list(curr_predictions[:left_num])

                    break

                start_index += batch_size

            alphabet = Alphabet()
            for i in range(num_classes):
                alphabet.add(str(i))
            confusionMatrix = ConfusionMatrix(alphabet)
            predictions = list(map(str, predictions))
            golds = list(map(str, golds))
            confusionMatrix.add_list(predictions, golds)

            return confusionMatrix


        def test_step_for_cqa(s1_all, s2_all, y_all, tag):
            """
            Evaluates model on a dev/test set
            """
            golds = []
            preds = []
            softmax_scores = []

            n = len(s1_all)
            batch_size = FLAGS.batch_size
            start_index = 0
            while start_index < n:
                if start_index + batch_size <= n:
                    s1_batch = s1_all[start_index: start_index + batch_size]
                    s2_batch = s2_all[start_index: start_index + batch_size]
                    y_batch = y_all[start_index: start_index + batch_size]

                    feed_dict = {
                        model.input_s1: s1_batch,
                        model.input_s2: s2_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: 1.0
                    }

                    step, loss, accuracy, curr_softmax_scores, curr_predictions, curr_golds = sess.run(
                        [global_step, model.loss, model.accuracy, model.softmax_scores,
                         model.predictions, model.golds], feed_dict)

                    golds += list(curr_golds)
                    preds += list(curr_predictions)
                    softmax_scores += list(curr_softmax_scores)

                else:
                    left_num = n - start_index
                    # 填充一下
                    s1_batch = np.concatenate((s1_all[start_index:], s1_all[:batch_size - left_num]), axis=0)
                    s2_batch = np.concatenate((s2_all[start_index:], s2_all[:batch_size - left_num]), axis=0)
                    y_batch = np.concatenate((y_all[start_index:], y_all[:batch_size - left_num]), axis=0)

                    feed_dict = {
                        model.input_s1: s1_batch,
                        model.input_s2: s2_batch,
                        model.input_y: y_batch,
                        model.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy, curr_softmax_scores, curr_predictions, curr_golds = sess.run(
                        [global_step, model.loss, model.accuracy, model.softmax_scores,
                         model.predictions, model.golds], feed_dict)

                    golds += list(curr_golds[:left_num])
                    preds += list(curr_predictions[:left_num])
                    softmax_scores += list(curr_softmax_scores[:left_num])

                    break

                start_index += batch_size

            alphabet = Alphabet()
            for i in range(num_classes):
                alphabet.add(str(i))
            confusionMatrix = ConfusionMatrix(alphabet)
            predictions = list(map(str, preds))
            golds = list(map(str, golds))
            confusionMatrix.add_list(predictions, golds)



            id_file = ""
            if tag == "dev":
                id_file = train_data_dir + "/dev/id"
            if tag == "test":
                id_file = train_data_dir + "/test/id"

            subtask = ""
            if train_data_dir.split("/")[-1] == "QA":
                subtask = "A"
            if train_data_dir.split("/")[-1] == "QQ":
                subtask = "B"

            pred_file = train_data_dir + "/result.%s.txt" % (timestamp)
            with open(pred_file, "w") as fw:
                for i, s in enumerate(softmax_scores):
                    fw.write("%d\t%.4f\n" % (preds[i], s[num_classes - 1]))

            print(pred_file, id_file, tag, subtask)
            map_score, mrr_score = get_rank_score_by_file(pred_file, id_file, tag, subtask)

            return map_score, mrr_score, confusionMatrix.get_accuracy()



        def _binary_early_stop(best_score, best_output_string):

            confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels)
            p, r, dev_f1 = confusionMatrix.get_prf("1")

            # current performance
            curr_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

            flag = 0
            if dev_f1 >= best_score:
                flag = 1
                best_score = dev_f1

                best_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

                # print("")
                # print("\nEvaluation on Test:")
                # confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels)
                # confusionMatrix.print_out()
                # print("")

                acc = confusionMatrix.get_accuracy()
                p, r, f1 = confusionMatrix.get_prf("1")

                evaluation_result["acc"] = "%.4f" % acc
                evaluation_result["f1"] = "%.4f" % f1
                evaluation_result["p"] = "%.4f" % p
                evaluation_result["r"] = "%.4f" % r

            color = colored.bg('black') + colored.fg('green')
            reset = colored.attr('reset')

            print("")
            print("\nEvaluation on Test:")
            print("Current Performance:")
            print(curr_output_string)
            if flag == 1:
                print("  " * 40 + '❤️')
            print((color + 'Best Performance' + reset))
            print((color + best_output_string + reset))
            print("")

            return best_score, best_output_string


        def _four_way_early_stop(best_score, best_output_string):

            confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels)

            acc = confusionMatrix.get_accuracy()
            # 对着 f1 调
            p, r, f1 = confusionMatrix.get_average_prf()

            # current performance
            curr_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

            flag = 0
            if f1 >= best_score:
                flag = 1
                best_score = f1

                best_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

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

            color = colored.bg('black') + colored.fg('green')
            reset = colored.attr('reset')

            print("")
            print("\nEvaluation on Test:")
            print("Current Performance:")
            print(curr_output_string)
            if flag == 1:
                print("  " * 40 + '❤️')
            print((color + 'Best Performance' + reset))
            print((color + best_output_string + reset))
            print("")

            return best_score, best_output_string


        def _conll_early_stop(best_score, best_output_string):

            confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels)

            # 对着 acc 调
            acc = confusionMatrix.get_accuracy()
            p, r, f1 = confusionMatrix.get_average_prf()

            # current performance
            curr_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

            flag = 0
            if acc >= best_score:
                flag = 1
                best_score = acc

                best_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

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

            color = colored.bg('black') + colored.fg('green')
            reset = colored.attr('reset')

            print("")
            print("\nEvaluation on Test:")
            print("Current Performance:")
            print(curr_output_string)
            if flag == 1:
                print("  " * 40 + '❤️')
            print((color + 'Best Performance' + reset))
            print((color + best_output_string + reset))
            print("")

            return best_score, best_output_string


        def _cqa_early_stop(best_score, best_output_string):

            map_score, mrr_score, acc = test_step_for_cqa(test_arg1s, test_arg2s, test_labels, tag="test")
            curr_output_string = "==> MAP: %f" % map_score

            flag = 0
            if map_score >= best_score:
                flag = 1
                best_score = map_score

                best_output_string = "==> MAP: %f" % best_score

                # print("")
                # print("\nEvaluation on Test:")
                # confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels)
                # confusionMatrix.print_out()
                # print("")

                evaluation_result["acc"] = "%.4f" % acc
                evaluation_result["f1"] = "%.4f" % map_score
                # evaluation_result["p"] = "%.4f" % p
                # evaluation_result["r"] = "%.4f" % r

            color = colored.bg('black') + colored.fg('green')
            reset = colored.attr('reset')

            print("")
            print("\nEvaluation on Test:")
            print("Current Performance:")
            print(curr_output_string)
            if flag == 1:
                print("  " * 40 + '❤️')
            print((color + 'Best Performance' + reset))
            print((color + best_output_string + reset))
            print("")

            return best_score, best_output_string


        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(train_arg1s, train_arg2s, train_labels)), FLAGS.batch_size, FLAGS.num_epochs)

        best_score = 0.0
        best_output_string = ""
        # Training loop. For each batch...
        for batch in batches:
            s1_batch, s2_batch, y_batch = list(zip(*batch))
            train_step(s1_batch, s2_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:

                if num_classes == 3:
                    # cqa
                    best_score, best_output_string = _cqa_early_stop(best_score, best_output_string)
                if num_classes == 2:
                    best_score, best_output_string = _binary_early_stop(best_score, best_output_string)
                if num_classes == 4:
                    best_score, best_output_string = _four_way_early_stop(best_score, best_output_string)
                if num_classes in [10, 15]:
                    best_score, best_output_string = _conll_early_stop(best_score, best_output_string)


# record the configuration and result
fieldnames = ["f1", "p", "r", "acc", "train_data_dir", "model", "share_rep_weights",
                   "bidirectional", "cell_type",  "hidden_size", "num_layers",
                  "dropout_keep_prob", "l2_reg_lambda", "Optimizer", "learning_rate", "batch_size", "num_epochs", "w2v_type",
                  "additional_conf"
]
do_record(fieldnames, configuration, additional_conf, evaluation_result, record_file)



