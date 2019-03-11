# -*- coding:utf-8 -*-
# https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "/home/dejian/data/ELMO/elmo_options.json"
weight_file = "/home/dejian/data/ELMO/elmo_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['I', 'love', 'you', '.'], ['Sorry', ',', 'I', 'don', "'t", 'love', 'you', '.'],['hhh','lidejian']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)

vector=embeddings['elmo_representations']

print(embeddings)
print("success!")

# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector