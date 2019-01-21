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
import tensorflow_fold as td

data_dir = '/home/dejian/data/glove.840B.300d'
print('saving files to %s' % data_dir)


def download_and_unzip(url_base, zip_name, *file_names):
    zip_path = os.path.join(data_dir, zip_name)
    url = url_base + zip_name
    print('downloading %s to %s' % (url, zip_path))
    urllib.request.urlretrieve(url, zip_path)
    out_paths = []
    with zipfile.ZipFile(zip_path, 'r') as f:
        for file_name in file_names:
            print('extracting %s' % file_name)
            out_paths.append(f.extract(file_name, path=data_dir))
    return out_paths


full_glove_path = '/home/dejian/data/glove.840B.300d/glove.840B.300d.txt'

train_path = '/home/dejian/data/glove.840B.300d/trees/train.txt'
dev_path = '/home/dejian/data/glove.840B.300d/trees/dev.txt'
test_path = '/home/dejian/data/glove.840B.300d/trees/test.txt'

filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')


def filter_glove():
    vocab = set()
    # Download the full set of unlabeled sentences separated by '|'.
    sentence_path, = download_and_unzip(
        'http://nlp.stanford.edu/~socherr/', 'stanfordSentimentTreebank.zip',
        'stanfordSentimentTreebank/SOStr.txt')
    with codecs.open(sentence_path, encoding='utf-8') as f:
        for line in f:
            # Drop the trailing newline and strip backslashes. Split into words.
            vocab.update(line.strip().replace('\\', '').split('|'))
    nread = 0
    nwrote = 0
    with codecs.open(full_glove_path, encoding='utf-8') as f:
        with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
            for line in f:
                nread += 1
                line = line.strip()
                if not line: continue
                if line.split(u' ', 1)[0] in vocab:
                    out.write(line + '\n')
                    nwrote += 1
    print('read %s lines, wrote %s' % (nread, nwrote))


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
# weigit_matrix  ndarray[20762,300]
# word_idx   dict (20725)  {'ask':5338...}

# Finally, load the treebank data.
def load_trees(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        # Drop the trailing newline and strip \s.
        trees = [line.strip().replace('\\', '') for line in f]
        print('loaded %s trees from %s' % (len(trees), filename))
        return trees


train_trees = load_trees(train_path)  # 得到源文件的内容，为一个列表，列表中每个元素为字符串，为源文件每一行内容
dev_trees = load_trees(dev_path)
test_trees = load_trees(test_path)


# '''Build the model'''

class BinaryTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    """LSTM with two state inputs.

    This is the model described in section 3.2 of 'Improved Semantic
    Representations From Tree-Structured Long Short-Term Memory
    Networks' <http://arxiv.org/pdf/1503.00075.pdf>, with recurrent
    dropout as described in 'Recurrent Dropout without Memory Loss'
    <http://arxiv.org/pdf/1603.05118.pdf>.
    """

    def __init__(self, num_units, keep_prob=1.0):
        """Initialize the cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          keep_prob: Keep probability for recurrent dropout.
        """
        super(BinaryTreeLSTMCell, self).__init__(num_units)
        self._keep_prob = keep_prob

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            lhs, rhs = state
            c0, h0 = lhs
            c1, h1 = rhs
            concat = tf.contrib.layers.linear(
                tf.concat([inputs, h0, h1], 1), 5 * self._num_units)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f0, f1, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            j = self._activation(j)
            if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
                j = tf.nn.dropout(j, self._keep_prob)

            new_c = (c0 * tf.sigmoid(f0 + self._forget_bias) +
                     c1 * tf.sigmoid(f1 + self._forget_bias) +
                     tf.sigmoid(i) * j)
            new_h = self._activation(new_c) * tf.sigmoid(o)

            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


# Use a placeholder for the dropout keep probability, with a default of 1 (for eval 为了评估).
keep_prob_ph = tf.placeholder_with_default(1.0, [])

# Create the LSTM cell for our model.
lstm_num_units = 300  # Tai et al. used 150, but our regularization strategy is more effective
tree_lstm = td.ScopedLayer(
    tf.contrib.rnn.DropoutWrapper(
        BinaryTreeLSTMCell(lstm_num_units, keep_prob=keep_prob_ph),
        input_keep_prob=keep_prob_ph, output_keep_prob=keep_prob_ph),
    name_or_scope='tree_lstm')

# Create the output layer using td.FC.
NUM_CLASSES = 5  # number of distinct sentiment labels
output_layer = td.FC(NUM_CLASSES, activation=None, name='output_layer')

# Create the word embedding using td.Embedding.。。。
word_embedding = td.Embedding(
    *weight_matrix.shape, initializer=weight_matrix, name='word_embedding')

embed_subtree = td.ForwardDeclaration(name='embed_subtree')


def logits_and_state():
    """Creates a block that goes from tokens to (logits, state) tuples."""
    unknown_idx = len(word_idx)
    lookup_word = lambda word: word_idx.get(word, unknown_idx)

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


# Define a per-node loss function for training.
def tf_node_loss(logits, labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)


def tf_fine_grained_hits(logits, labels):
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    return tf.cast(tf.equal(predictions, labels), tf.float64)


def tf_binary_hits(logits, labels):
    softmax = tf.nn.softmax(logits)
    binary_predictions = (softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1])
    binary_labels = labels > 2
    return tf.cast(tf.equal(binary_predictions, binary_labels), tf.float64)


def add_metrics(is_root, is_neutral):
    """A block that adds metrics for loss and hits; output is the LSTM state."""
    c = td.Composition(
        name='predict(is_root=%s, is_neutral=%s)' % (is_root, is_neutral))
    with c.scope():
        # destructure the input; (labels, (logits, state))
        labels = c.input[0]
        logits = td.GetItem(0).reads(c.input[1])
        state = td.GetItem(1).reads(c.input[1])

        # calculate loss
        loss = td.Function(tf_node_loss)
        td.Metric('all_loss').reads(loss.reads(logits, labels))
        if is_root: td.Metric('root_loss').reads(loss)

        # calculate fine-grained hits
        hits = td.Function(tf_fine_grained_hits)
        td.Metric('all_hits').reads(hits.reads(logits, labels))
        if is_root: td.Metric('root_hits').reads(hits)

        # calculate binary hits, if the label is not neutral
        if not is_neutral:
            binary_hits = td.Function(tf_binary_hits).reads(logits, labels)
            td.Metric('all_binary_hits').reads(binary_hits)
            if is_root: td.Metric('root_binary_hits').reads(binary_hits)

        # output the state, which will be read by our by parent's LSTM cell
        c.output.reads(state)
    return c


def tokenize(s):
    label, phrase = s[1:-1].split(None, 1)
    return label, sexpr.sexpr_tokenize(phrase)


# Try it out.
# print(tokenize('(X Y)'))


def embed_tree(logits_and_state, is_root):
    """Creates a block that embeds trees; output is tree LSTM state."""
    return td.InputTransform(tokenize) >> td.OneOf(
        key_fn=lambda pair: pair[0] == '2',  # label 2 means neutral
        case_blocks=(add_metrics(is_root, is_neutral=False),
                     add_metrics(is_root, is_neutral=True)),
        pre_block=(td.Scalar('int32'), logits_and_state))


# Put everything together and create our top-level (i.e. root) model. It is rather simple.
model = embed_tree(logits_and_state(), is_root=True)

# 通过第二次调用embed_tree来解析嵌入子树的前向声明（不是根节点情况）。
embed_subtree.resolve_to(embed_tree(logits_and_state(), is_root=False))

# Compile the model.
compiler = td.Compiler.create(model)
print('input type: %s' % model.input_type)
print('output type: %s' % model.output_type)

# '''Setup for training'''
# Calculate means by summing the raw metrics.
metrics = {k: tf.reduce_mean(v) for k, v in compiler.metric_tensors.items()}

# Magic numbers.
LEARNING_RATE = 0.05
KEEP_PROB = 0.75
BATCH_SIZE = 100
EPOCHS = 20
EMBEDDING_LEARNING_RATE_FACTOR = 0.1

# Training with Adagrad.
train_feed_dict = {keep_prob_ph: KEEP_PROB}
loss = tf.reduce_sum(compiler.metric_tensors['all_loss'])
opt = tf.train.AdagradOptimizer(LEARNING_RATE)

# downscale the gradients for the word embedding vectors 10x 以防止过拟合
grads_and_vars = opt.compute_gradients(loss)
found = 0
for i, (grad, var) in enumerate(grads_and_vars):    # grads_and_vars:list,0-4,每个里面是是个tuple,('tensorflow.python.framework.ops.Tensor',  'tensorflow.python.ops.variables.Variable')
    if var == word_embedding.weights:
        found += 1
        grad = tf.scalar_mul(EMBEDDING_LEARNING_RATE_FACTOR, grad)
        grads_and_vars[i] = (grad, var)
assert found == 1  # internal consistency check
train = opt.apply_gradients(grads_and_vars)
saver = tf.train.Saver()

# The TF graph is now complete; initialize the variables.
sess.run(tf.global_variables_initializer())


# '''Train the model'''

# Start by defining a function that does a single step of training on a batch and returns the loss.
def train_step(batch):
    train_feed_dict[compiler.loom_input_tensor] = batch
    _, batch_loss = sess.run([train, loss], train_feed_dict)
    return batch_loss


# Now similarly for an entire epoch of training.
def train_epoch(train_set):
    return sum(train_step(batch) for batch in td.group_by_batches(train_set, BATCH_SIZE))


# Use Compiler.build_loom_inputs() to transform train_trees into individual loom inputs (i.e. wiring diagrams) that\
# we can use to actually run the model.
train_set = compiler.build_loom_inputs(train_trees) # map object

# Use Compiler.build_feed_dict() to build a feed dictionary for validation on the dev set.
dev_feed_dict = compiler.build_feed_dict(dev_trees) # dict ,len=1, key的type: 'tensorflow.python.framework.ops.Tensor', value:'tensorflow_fold.blocks.util.EdibleIterator'


# Define a function to do an eval on the dev set and pretty-print some stats, returning accuracy on the dev set.
def dev_eval(epoch, train_loss):
    dev_metrics = sess.run(metrics, dev_feed_dict)
    dev_loss = dev_metrics['all_loss']
    dev_accuracy = ['%s: %.2f' % (k, v * 100) for k, v in
                    sorted(dev_metrics.items()) if k.endswith('hits')]
    print('epoch:%4d, train_loss: %.3e, dev_loss_avg: %.3e, dev_accuracy:\n  [%s]'
          % (epoch, train_loss, dev_loss, ' '.join(dev_accuracy)))
    return dev_metrics['root_hits']


# Run the main training loop, saving the model after each epoch if it has the best accuracy on the dev set.
# Use the td.epochs utility function to memoize the loom inputs and shuffle them after every epoch of training.
best_accuracy = 0.0
save_path = os.path.join(data_dir, 'sentiment_model')
for epoch, shuffled in enumerate(td.epochs(train_set, EPOCHS), 1):
    train_loss = train_epoch(shuffled)
    accuracy = dev_eval(epoch, train_loss)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        checkpoint_path = saver.save(sess, save_path, global_step=epoch)
        print('model saved in file: %s' % checkpoint_path)

# '''Evaluate the model'''
saver.restore(sess, checkpoint_path)
test_results = sorted(sess.run(metrics, compiler.build_feed_dict(test_trees)).items())
print('    loss: [%s]' % ' '.join(
    '%s: %.3e' % (name.rsplit('_', 1)[0], v)
    for name, v in test_results if name.endswith('_loss')))
print('accuracy: [%s]' % ' '.join(
    '%s: %.2f' % (name.rsplit('_', 1)[0], v * 100)
    for name, v in test_results if name.endswith('_hits')))


'''
ssh://dejian@49.52.10.181:22/home/dejian/.conda/envs/tensorflow/bin/python -u /home/dejian/home/dejian/pycharm_space/TreeLSTM.py
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
saving files to /home/dejian/data/glove.840B.300d
loading word embeddings from /home/dejian/data/glove.840B.300d/filtered_glove.txt
loaded 8544 trees from /home/dejian/data/glove.840B.300d/trees/train.txt
loaded 1101 trees from /home/dejian/data/glove.840B.300d/trees/dev.txt
loaded 2210 trees from /home/dejian/data/glove.840B.300d/trees/test.txt
('X', ['Y'])
input type: PyObjectType()
output type: TupleType(TensorType((300,), 'float32'), TensorType((300,), 'float32'))
/home/dejian/.conda/envs/tensorflow/lib/python3.5/site-packages/tensorflow/python/ops/gradients_impl.py:91: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
epoch:   1, train_loss: 2.308e+05, dev_loss_avg: 5.275e-01, dev_accuracy:
  [all_binary_hits: 88.30 all_hits: 78.27 root_binary_hits: 79.82 root_hits: 41.78]
model saved in file: /home/dejian/data/glove.840B.300d/sentiment_model-1
epoch:   2, train_loss: 1.605e+05, dev_loss_avg: 4.615e-01, dev_accuracy:
  [all_binary_hits: 90.33 all_hits: 81.00 root_binary_hits: 84.29 root_hits: 47.32]
model saved in file: /home/dejian/data/glove.840B.300d/sentiment_model-2
epoch:   3, train_loss: 1.454e+05, dev_loss_avg: 4.389e-01, dev_accuracy:
  [all_binary_hits: 91.31 all_hits: 81.84 root_binary_hits: 86.12 root_hits: 49.32]
model saved in file: /home/dejian/data/glove.840B.300d/sentiment_model-3
epoch:   4, train_loss: 1.366e+05, dev_loss_avg: 4.262e-01, dev_accuracy:
  [all_binary_hits: 91.77 all_hits: 82.40 root_binary_hits: 86.81 root_hits: 49.05]
epoch:   5, train_loss: 1.303e+05, dev_loss_avg: 4.244e-01, dev_accuracy:
  [all_binary_hits: 91.90 all_hits: 82.52 root_binary_hits: 87.27 root_hits: 50.95]
model saved in file: /home/dejian/data/glove.840B.300d/sentiment_model-5
epoch:   6, train_loss: 1.254e+05, dev_loss_avg: 4.158e-01, dev_accuracy:
  [all_binary_hits: 92.24 all_hits: 82.91 root_binary_hits: 87.84 root_hits: 50.86]
epoch:   7, train_loss: 1.213e+05, dev_loss_avg: 4.138e-01, dev_accuracy:
  [all_binary_hits: 92.35 all_hits: 82.94 root_binary_hits: 88.30 root_hits: 50.95]
epoch:   8, train_loss: 1.179e+05, dev_loss_avg: 4.144e-01, dev_accuracy:
  [all_binary_hits: 92.24 all_hits: 82.89 root_binary_hits: 87.16 root_hits: 52.04]
model saved in file: /home/dejian/data/glove.840B.300d/sentiment_model-8
epoch:   9, train_loss: 1.150e+05, dev_loss_avg: 4.149e-01, dev_accuracy:
  [all_binary_hits: 92.35 all_hits: 82.91 root_binary_hits: 87.73 root_hits: 52.68]
model saved in file: /home/dejian/data/glove.840B.300d/sentiment_model-9
epoch:  10, train_loss: 1.122e+05, dev_loss_avg: 4.143e-01, dev_accuracy:
  [all_binary_hits: 92.40 all_hits: 82.96 root_binary_hits: 87.50 root_hits: 51.23]
epoch:  11, train_loss: 1.103e+05, dev_loss_avg: 4.164e-01, dev_accuracy:
  [all_binary_hits: 92.40 all_hits: 82.99 root_binary_hits: 87.61 root_hits: 51.41]
epoch:  12, train_loss: 1.079e+05, dev_loss_avg: 4.201e-01, dev_accuracy:
  [all_binary_hits: 92.00 all_hits: 83.06 root_binary_hits: 86.58 root_hits: 52.41]
epoch:  13, train_loss: 1.059e+05, dev_loss_avg: 4.216e-01, dev_accuracy:
  [all_binary_hits: 92.07 all_hits: 83.07 root_binary_hits: 87.04 root_hits: 51.41]
epoch:  14, train_loss: 1.040e+05, dev_loss_avg: 4.242e-01, dev_accuracy:
  [all_binary_hits: 91.90 all_hits: 83.06 root_binary_hits: 86.93 root_hits: 52.23]
epoch:  15, train_loss: 1.022e+05, dev_loss_avg: 4.283e-01, dev_accuracy:
  [all_binary_hits: 91.83 all_hits: 82.97 root_binary_hits: 86.35 root_hits: 50.86]
epoch:  16, train_loss: 1.006e+05, dev_loss_avg: 4.285e-01, dev_accuracy:
  [all_binary_hits: 92.12 all_hits: 83.04 root_binary_hits: 87.04 root_hits: 51.41]
epoch:  17, train_loss: 9.917e+04, dev_loss_avg: 4.337e-01, dev_accuracy:
  [all_binary_hits: 92.02 all_hits: 82.98 root_binary_hits: 86.93 root_hits: 50.77]
epoch:  18, train_loss: 9.808e+04, dev_loss_avg: 4.370e-01, dev_accuracy:
  [all_binary_hits: 92.15 all_hits: 82.99 root_binary_hits: 87.84 root_hits: 51.68]
epoch:  19, train_loss: 9.661e+04, dev_loss_avg: 4.399e-01, dev_accuracy:
  [all_binary_hits: 91.90 all_hits: 82.96 root_binary_hits: 87.04 root_hits: 52.41]
epoch:  20, train_loss: 9.517e+04, dev_loss_avg: 4.379e-01, dev_accuracy:
  [all_binary_hits: 92.09 all_hits: 82.83 root_binary_hits: 87.84 root_hits: 49.14]
    loss: [all: 4.165e-01 root: 1.088e+00]
accuracy: [all_binary: 92.25 all: 82.91 root_binary: 89.13 root: 52.26]
'''