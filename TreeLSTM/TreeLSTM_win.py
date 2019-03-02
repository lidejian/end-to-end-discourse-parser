# -*- coding:utf-8 -*-
# boilerplate
import codecs
import functools
import os
import tempfile
import zipfile

from nltk.tokenize import sexpr
import numpy as np
from six.moves import urllib
import tensorflow as tf

sess = tf.InteractiveSession()
# import tensorflow_fold as td


# data_dir = "D:\data\glove.840B.300d"
# print('saving files to %s' % data_dir)


# def download_and_unzip(url_base, zip_name, *file_names):
#     zip_path = os.path.join(data_dir, zip_name)
#     url = url_base + zip_name
#     print('downloading %s to %s' % (url, zip_path))
#     urllib.request.urlretrieve(url, zip_path)
#     out_paths = []
#     with zipfile.ZipFile(zip_path, 'r') as f:
#         for file_name in file_names:
#             print('extracting %s' % file_name)
#             out_paths.append(f.extract(file_name, path=data_dir))
#     return out_paths


# full_glove_path = r"D:\data\glove.840B.300d\glove.840B.300d.txt"
train_path = r"D:\data\trainDevTestTrees_PTB\trees\train.txt"
dev_path = r"D:\data\trainDevTestTrees_PTB\trees\dev.txt"
test_path = r"D:\data\trainDevTestTrees_PTB\trees\test.txt"

# filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')
filtered_glove_path = r'D:\data\glove.840B.300d\filtered_glove.txt'



# def filter_glove():
#     vocab = set()
#     # Download the full set of unlabeled sentences separated by '|'.
#     sentence_path, = download_and_unzip(
#         'http://nlp.stanford.edu/~socherr/', 'stanfordSentimentTreebank.zip',
#         'stanfordSentimentTreebank/SOStr.txt')
#     with codecs.open(sentence_path, encoding='utf-8') as f:
#         for line in f:
#             # Drop the trailing newline and strip backslashes. Split into words.
#             vocab.update(line.strip().replace('\\', '').split('|'))
#     nread = 0
#     nwrote = 0
#     with codecs.open(full_glove_path, encoding='utf-8') as f:
#         with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
#             for line in f:
#                 nread += 1
#                 line = line.strip()
#                 if not line: continue
#                 if line.split(u' ', 1)[0] in vocab:
#                     out.write(line + '\n')
#                     nwrote += 1
#     print('read %s lines, wrote %s' % (nread, nwrote))
#
#
# filter_glove()


def load_embeddings(embedding_path):
    """Loads embedings, returns weight matrix and dict from words to indices."""
    print('loading word embeddings from %s' % embedding_path)
    weight_vectors = []
    word_idx = {}
    with codecs.open(embedding_path, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            word_idx[word] = len(weight_vectors)
            weight_vectors.append(np.array(vec.split(), dtype=np.float32))
    # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
    # '-RRB-' respectively in the parse-trees.
    word_idx[u'-LRB-'] = word_idx.pop(u'(')
    word_idx[u'-RRB-'] = word_idx.pop(u')')
    # Random embedding vector for unknown words.
    weight_vectors.append(np.random.uniform(
        -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
    return np.stack(weight_vectors), word_idx


weight_matrix, word_idx = load_embeddings(filtered_glove_path)


def load_trees(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        # Drop the trailing newline and strip \s.
        trees = [line.strip().replace('\\', '') for line in f]
        print('loaded %s trees from %s' % (len(trees), filename))
        return trees


train_trees = load_trees(train_path)
dev_trees = load_trees(dev_path)
test_trees = load_trees(test_path)

def logits_and_state():
    """Creates a block that goes from tokens to (logits, state) tuples."""
    unknown_idx = len(word_idx)
    lookup_word = lambda word: word_idx.get(word, unknown_idx)

    print("hhhhhh")

    word2vec = (td.GetItem(0) >> td.InputTransform(lookup_word) >>
                td.Scalar('int32') >> word_embedding)

    pair2vec = (embed_subtree(), embed_subtree())

    # Trees are binary, so the tree layer takes two states as its input_state.
    zero_state = td.Zeros((tree_lstm.state_size,) * 2)
    # Input is a word vector.
    zero_inp = td.Zeros(word_embedding.output_type.shape[0])

    word_case = td.AllOf(word2vec, zero_state)
    pair_case = td.AllOf(zero_inp, pair2vec)

    tree2vec = td.OneOf(len, [(1, word_case), (2, pair_case)])

    return tree2vec >> tree_lstm >> (output_layer, td.Identity())

logits_and_state()