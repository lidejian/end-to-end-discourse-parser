# -*- coding:utf-8 -*-
# https://cstsunfu.github.io/2018/06/ELMo/
from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder(options_file='/home/dejian/data/ELMO/elmo_options.json',
                    weight_file='/home/dejian/data/ELMO/elmo_weights.hdf5', cuda_device=0)
context_tokens = [['I', 'love', 'you', '.'], ['Sorry', ',', 'I', 'don', "'t", 'love', 'you', '.'],['hhh','lidejian'],['hhh','lidejian']]
elmo_embedding, elmo_mask = elmo.batch_to_embeddings(context_tokens)

vectors = elmo.embed_sentence(context_tokens[0])



print(elmo_embedding)
print(elmo_mask)
print("success!!!")


