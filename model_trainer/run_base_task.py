#!/usr/bin/env python
#encoding: utf-8
import sys
import imp
imp.reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("../")
import config
import os


# ''' binary '''
# # "PDTB_imp",
# # "PDTB_imp_and_PDTB_exp"
# # "PDTB_imp_and_BLLIP_exp"
# dataset_type = "PDTB_imp"
#
# "Comparison",
# "Contingency",
# "Expansion",
# "Temporal"
# level1_type = "Comparison"
# level1_type = "Contingency"
# level1_type = "Expansion"
# level1_type = "Temporal"
# level1_type = "ExpEntRel"
#
# train_data_dir = config.DATA_PATH + "/binary/%s/%s" % (dataset_type, level1_type)


# ''' four way'''
# train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"


# ''' sentense four eng'''
# train_data_dir = config.DATA_PATH + "/gen_my_four_sen/exp"
train_data_dir = config.DATA_PATH + "/gen_my_four_sen/imp"



# # # CoNLL
# train_data_dir = config.DATA_PATH + "/conll/conll_imp"
# blind = False


# train_data_dir = config.DATA_PATH + "/ZH"
# blind = True


# model = "CNN"
model = "RNN"
# model = "Attention_RNN1"
# model = "Attention_RNN2" # mine
# model = "Attention_RNN3"
# model = "Attention_RNN4"
# model = "Attention_RNN5"
share_rep_weights = False
bidirectional = False
embedding = "Glove" # Glove or google_word_2_vec
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

cmd = "python train_base_task.py" \
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
      + " --embedding %s" % embedding\

# + " --blind %s" % blind \
# + " --dataset_type %s" % dataset_type \
# + " --level1_sense %s" % level1_type \

print(cmd)
os.system(cmd)




