# encoding: utf-8
import codecs
import glob
import json

import numpy as np
import re, os
from allennlp.commands.elmo import ElmoEmbedder
import torch

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string


def load_data_and_labels(dir):

    with \
        codecs.open("%s/arg1" % dir, encoding="utf-8") as fin_arg1, \
        codecs.open("%s/arg2" % dir, encoding="utf-8") as fin_arg2, \
        codecs.open("%s/label" % dir, encoding="utf-8") as fin_label:

        arg1s = [clean_str(line.strip()) for line in fin_arg1]
        arg2s = [clean_str(line.strip()) for line in fin_arg2]

        arg1s = [s.lower() for s in arg1s]
        arg2s = [s.lower() for s in arg2s]

        labels = [int(x.strip()) for x in fin_label]
        num_classes = max(labels) + 1
        if num_classes == 14:
            num_classes += 1

        _labels = []
        for label in labels:
            l = [0] * num_classes
            l[label] = 1
            _labels.append(l)
        labels = np.array(_labels)
        return arg1s, arg2s, labels


def build_vocab(dir):
    vocab = {}
    for path in glob.glob(os.path.join(dir, '*/*.tok')):
        with open(path) as fin:
            for line in fin:
                line = clean_str(line.strip())
                for token in line.split(" "):
                    vocab.add(token)

    for token in [] + sorted(vocab):
        pass






def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """

    np.random.seed(1)

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            # end_index = min((batch_num + 1) * batch_size, data_size)

            # 不够的时候，补齐！
            if (batch_num + 1) * batch_size > data_size:

                yield np.concatenate((shuffled_data[start_index:], shuffled_data[:(batch_num + 1) * batch_size - data_size]), axis=0)

            else:
                end_index = (batch_num + 1) * batch_size
                yield shuffled_data[start_index:end_index]

def extracting_embedding_from_elmo(arg, max_length):
    elmo = ElmoEmbedder(options_file='/home/dejian/data/ELMO/elmo_options.json',
                        weight_file='/home/dejian/data/ELMO/elmo_weights.hdf5', cuda_device=0)
    arg_token = [sen.split(" ")[:max_length] for sen in arg]
    arg_embedding, elmo_mask = elmo.batch_to_embeddings(arg_token)
    torch.cuda.empty_cache()
    arg_embed = arg_embedding.cpu().numpy()[:, 1, :, :]
    return arg_embed

if __name__ == '__main__':
    a = [[0, 1], [0, 1], [1, 0]]
    b = [[1, 0], [1, 1], [1, 0]]
    print(np.concatenate([a, b], 0))
