# -*- coding:utf-8 -*-
from allennlp.commands.elmo import ElmoEmbedder

options_file = "/home/dejian/data/ELMO/elmo_options.json"
weight_file = "/home/dejian/data/ELMO/elmo_weights.hdf5"

elmo = ElmoEmbedder(options_file='/home/dejian/data/ELMO/elmo_options.json',
                    weight_file='/home/dejian/data/ELMO/elmo_weights.hdf5', cuda_device=0)
tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
vectors = elmo.embed_sentence(tokens)
print(vectors)
print('success')