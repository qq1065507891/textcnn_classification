# encoding:utf-8
import tensorflow as tf


class TCNNConfig():
    embedding_size = 100  # dimension of word embedding
    vocab_size = 10000  # number of vocabulary
    pre_training = None  # use vector_char trained by word2vec

    seq_length = 60  # max length of sentence
    num_classes = 2  # number of labels
    num_filters = 128  # number of convolution kernel
    filter_sizes = [2, 3, 4]  # size of convolution kernel
    keep_prob = 0.5  # droppout
    lr = 1e-3  # learning rate
    lr_decay = 0.9  # learning rate decay
    clip = 6.0  # gradient clipping threshold
    l2_reg_lambda = 0.01  # l2 regularization lambda
    num_epochs = 10  # epochs
    batch_size = 64  # batch_size
    print_per_batch = 100  # print result
    save_per_batch = 10  # 每多少轮存入tensorboard
    train_filename = './train.txt'  # train data
    test_filename = './test.txt'  # test data
    vocab_filename = './vocab.txt'  # vocabulary
    vector_word_filename = './vector_word.txt'  # vector_word trained by word2vec
    vector_word_npz = './vector_word.npz'  # save vector_word to numpy file


class TextCNN(object):

    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)

        self.cnn()

    def cnn(self, attention_dim=100, use_attention=True):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_training))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            if use_attention:
                self.attention_hidden_dim = attention_dim
                self.attention_W = tf.Variable(
                    tf.random_uniform([self.config.embedding_size, self.attention_hidden_dim], 0.0, 1.0),
                    name="attention_W")
                self.attention_U = tf.Variable(
                    tf.random_uniform([self.config.embedding_size, self.attention_hidden_dim], 0.0, 1.0),
                    name="attention_U")
                self.attention_V = tf.Variable(tf.random_uniform([self.attention_hidden_dim, 1], 0.0, 1.0),
                                               name="attention_V")
                # attention layer before convolution
                self.output_att = list()
                with tf.name_scope("attention"):
                    input_att = tf.split(self.embedding_inputs, self.config.seq_length, axis=1)
                    for index, x_i in enumerate(input_att):
                        x_i = tf.reshape(x_i, [-1, self.config.embedding_size])
                        c_i = self.attention(x_i, input_att, index)
                        inp = tf.concat([x_i, c_i], axis=1)
                        self.output_att.append(inp)

                    input_conv = tf.reshape(tf.concat(self.output_att, axis=1),
                                            [-1, self.config.seq_length, self.config.embedding_size * 2],
                                            name="input_convolution")
                self.input_conv_expanded = tf.expand_dims(input_conv, -1)
            else:
                self.input_conv_expanded = tf.expand_dims(self.embedding_inputs, -1)

            self.dim_input_conv = self.input_conv_expanded.shape[-2].value
        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.dim_input_conv, 1, self.config.num_filters]
                    W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b1 = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                    conv1 = tf.nn.conv2d(
                        self.input_conv_expanded,
                        W1,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu")
                    pooled1 = tf.nn.max_pool(
                        h1,
                        ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled1)

            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.outputs = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.final_output = tf.nn.dropout(self.outputs, self.keep_prob)

        with tf.name_scope('output'):
            fc_w = tf.get_variable('fc_w', shape=[self.final_output.shape[1].value, self.config.num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.l2_loss += tf.nn.l2_loss(fc_w)
            self.l2_loss += tf.nn.l2_loss(fc_b)
            self.loss = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda * self.l2_loss
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def attention(self, x_i, x, index):
        """
        Attention model for Neural Machine Translation
        :param x_i: the embedded input at time i
        :param x: the embedded input of all times(x_j of attentions)
        :param index: step of time
        """

        e_i = []
        c_i = []
        for output in x:
            output = tf.reshape(output, [-1, self.config.embedding_size])
            atten_hidden = tf.tanh(
                tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U)))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j)
        e_i = tf.concat(e_i, axis=1)
        # e_i = tf.exp(e_i)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(alpha_i, self.config.seq_length, 1)

        # i!=j
        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = tf.reshape(output, [-1, self.config.embedding_size])
                c_i_j = tf.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.config.seq_length - 1, self.config.embedding_size])
        c_i = tf.reduce_sum(c_i, 1)
        return c_i
