#!/usr/bin/env python
#encoding: utf-8
import sys
import imp
imp.reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("../")
import config
import os



# bert-as-service
# https://github.com/hanxiao/bert-as-service
# start service:
# bert-serving-start -max_seq_len 100 -pooling_strategy NONE -model_dir /home/dejian/bert/uncased_L-12_H-768_A-12/  -num_worker=4

# bert-serving-start -pooling_strategy NONE -model_dir /home/dejian/bert/uncased_L-12_H-768_A-12/ -tuned_model_dir /home/dejian/pycharm_space/bert_linux/tmp/pdtb_output/ -num_worker=4



# ''' sentense four eng'''
# train_data_dir = config.DATA_PATH + "/gen_my_four_sen/exp"
train_data_dir = config.DATA_PATH + "/gen_my_four_sen/imp"



# # # CoNLL
# train_data_dir = config.DATA_PATH + "/conll/conll_imp"
# blind = False


# train_data_dir = config.DATA_PATH + "/ZH"
# blind = True


model = "CNN"
# model = "RNN"
# model = "Attention_RNN1"
# model = "Attention_RNN2" # mine
# model = "Attention_RNN3"
# model = "Attention_RNN4"
# model = "Attention_RNN5"
share_rep_weights = False
bidirectional = False
cell_type = "BasicLSTM"
# cell_type = "TreeLSTM"
hidden_size = 50
num_layers = 1
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
learning_rate = 0.001
batch_size = 64
num_epochs = 20
evaluate_every = 10

cmd = "python train_bert_task.py" \
      + " --train_data_dir %s" % train_data_dir \
      + " --model %s" % model \
      + " --share_rep_weights %s" % share_rep_weights \
      + " --bidirectional %s" % bidirectional \
      + " --cell_type %s" % cell_type \
      + " --hidden_size %s" % hidden_size \
      + " --num_layers %s" % num_layers \
      + " --dropout_keep_prob %s" % dropout_keep_prob \
      + " --l2_reg_lambda %s" % l2_reg_lambda \
      + " --learning_rate %s" % learning_rate \
      + " --batch_size %s" % batch_size \
      + " --num_epochs %s" % num_epochs \
      + " --evaluate_every %s" % evaluate_every \


# + " --blind %s" % blind \
# + " --dataset_type %s" % dataset_type \
# + " --level1_sense %s" % level1_type \

print(cmd)
os.system(cmd)




