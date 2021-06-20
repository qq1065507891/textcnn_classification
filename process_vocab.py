import sys
from collections import Counter

import numpy as np
import tensorflow.keras as kr
import codecs


def build_vocab(train_path, vocab_path, vocab_size=10000):
    """根据训练集构建词汇表，存储"""
    input = []
    target = []
    all_words = []
    for line in open(train_path, encoding='utf-8').readlines():
        temp = line.split("__label__")
        input.append(temp[0].strip())
        target.append(temp[1].strip())
    for sentence in input:
        sentence = sentence.split()
        all_words.extend(sentence)
    counter = Counter(all_words)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(vocab_path, mode='a', encoding='utf-8', errors='ignore').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open(vocab_dir, encoding='utf-8').read().strip().split()
    # print(words)
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


# def read_category():
#     """
#     Args:
#         None
#     Returns:
#         categories: a list of label
#         cat_to_id: a dict of label to id
#
#     """
#     categories = ['0', '1']
#     cat_to_id = dict(zip(categories, range(len(categories))))
#     return categories, cat_to_id


def process_file(filename, word_to_id, max_length=60):
    """将文件转换为id表示"""
    input = []
    target = []
    for line in open(filename, encoding='utf-8').readlines():
        temp = line.split("__label__")
        input.append(temp[0].strip())
        target.append(temp[1].strip())
    data_id = []
    for i in range(len(input)):
        data_id.append([word_to_id[x] for x in input[i] if x in word_to_id])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(target, num_classes=2)  # 将标签转换为one-hot表示
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = np.array(y)[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def export_word2vec_vectors(vocab, word2vec_dir, trimmed_filename):
    """
    Args:
        vocab: word_to_id
        word2vec_dir:file path of have trained word vector by word2vec
        trimmed_filename:file path of changing word_vector to numpy file
    Returns:
        save vocab_vector to numpy file

    """
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]
