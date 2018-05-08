import codecs
import itertools
import time

import numpy as np
import pickle
np.random.seed(133)


def get_id(corpus, key):
    """
    Get the corpus ID of a word
    (handle utf-8 encoding errors, following change
    https://github.com/vered1986/LexNET/pull/2 from @gossebouma)
    :param corpus: the corpus' resource object
    :param key: the word
    :return: the ID of the word or -1 if not found
    """
    id = -1  # the index of the unknown word
    try:
        id = corpus.get_id_by_term(key.encode('utf-8'))
    except UnicodeEncodeError:
        pass
    return id


def vectorize_path(path, lemma_index, pos_index, dep_index, dir_index):
    """
    Return a vector representation of the path
    :param path: the string representation of a path
    :param lemma_index: index to lemma dictionary
    :param pos_index: index to POS dictionary
    :param dep_index: index to dependency label dictionary
    :param dir_index: index to edge direction dictionary
    :return:
    """
    path_edges = [vectorize_edge(edge, lemma_index, pos_index, dep_index, dir_index) for edge in path.split('_')]

    if None in path_edges:
        return None

    return tuple(path_edges)


def vectorize_edge(edge, lemma_index, pos_index, dep_index, dir_index):
    """
    Return a vector representation of the edge: concatenate lemma/pos/dep and add direction symbols
    :param edge: the string representation of an edge
    :param lemma_index: index to lemma dictionary
    :param pos_index: index to POS dictionary
    :param dep_index: index to dependency label dictionary
    :param dir_index: index to edge direction dictionary
    :return:
    """
    try:
        lemma, pos, dep, direction = edge.split('/')
        lemma, pos, dep, direction = lemma_index.get(lemma, 0), pos_index[pos], dep_index[dep], dir_index[direction]
    except:
        return None

    return tuple([lemma, pos, dep, direction])


def reconstruct_edge((lemma, pos, dep, direction),
                     lemma_inverted_index, pos_inverted_index, dep_inverted_index, dir_inverted_index):
    """
    Return a string representation of the edge
    :param lemma_inverted_index: lemma to index dictionary
    :param pos_inverted_index: POS to index dictionary
    :param dep_inverted_index: dependency label to index dictionary
    :param dir_inverted_index: edge direction to index dictionary
    :return: The string representation of the edge
    """
    edge = '/'.join([lemma_inverted_index[lemma], pos_inverted_index[pos], dep_inverted_index[dep],
                     dir_inverted_index[direction]])
    return edge


def load_embeddings(file_name, path_lemmas, x_y_words, debug):
    """
    Load the pre-trained embeddings from a file
    :param file_name: the embeddings file
    :param vocabulary: limited vocabulary to load vectors for
    :return: the vocabulary and the word vectors
    """
    # if os.path.exists('{}.ebd_info'.format(file_name)):
    #     (wv, word_index, word_set) = pickle.load(open('{}.ebd_info'.format(file_name), 'rb'))
    #     return wv, word_index, word_set
    vocabulary = path_lemmas.union(x_y_words)
    print 'time before reading ebd', time.ctime()
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        # next(f_in)
        if debug and False:
            words = ['tmp']
            wv = np.random.rand(1, 50)
        else:
            if 'w2v' in file_name:
                f_in.readline()
            words, vectors = zip(*[line.strip().split(' ', 1) for line in f_in])
            wv = np.loadtxt(vectors)
    print 'time after reading ebd', time.ctime()
    # Add the unknown words
    unknown_vector = np.random.random_sample((wv.shape[1],))

    word_set = set(words)
    unknown_words = list(set(vocabulary).difference(set(words)))
    word_index = {w: i for i, w in enumerate(words)}
    # Create vectors for MWEs - sum of word embeddings, and OOV words
    # unknown_word_vectors = [np.add.reduce([wv[words.index(w)] if w in word_set else unknown_vector
    #                                        for w in word.split(' ')])
    #                         for word in unknown_words]
    word_len_l = []
    word_has_ebd_l = []
    for word in unknown_words:
        word_len = 0
        word_has_ebd = 0
        for w in word.split('_'):
            word_len += 1
            if w in word_set:
                word_has_ebd += 1
        word_len_l.append(word_len)
        word_has_ebd_l.append(word_has_ebd)
    print '% of words in phrases that have ebd', np.mean(
        np.array(word_has_ebd_l) / np.array(word_len_l, dtype=np.float))
    # if not debug:
    #     pickle.dump((unknown_words, word_len_l, word_has_ebd_l), open('ebd_check.pkl', 'wb'))

    unknown_word_vectors = [np.add.reduce([wv[word_index[w]] if w in word_set else unknown_vector
                                           for w in word.split('_')])
                            for word in unknown_words]
    # words_new = []
    # vocabulary = list(vocabulary)
    # for i in range(len(vocabulary)):
    #     if vocabulary[i] in word_index:
    #         words_new.append(vocabulary[i])
    # vectors_new = np.zeros((len(words_new), wv.shape[1]))
    # for i in range(len(words_new)):
    #     vectors_new[i] = wv[word_index[words_new[i]]]
    wv = np.vstack((wv, unknown_word_vectors))
    words = list(words) + unknown_words
    # wv = np.vstack((vectors_new, unknown_word_vectors))
    # words = words_new + unknown_words
    # print 'new len(words)=', len(words)
    print 'Known path_lemmas:', len(path_lemmas) - len(path_lemmas - word_set), '/', len(path_lemmas)
    print 'Known x_y_words:', len(x_y_words) - len(x_y_words - word_set), '/', len(x_y_words)
    print 'Known lemmas:', len(vocabulary) - len(unknown_words), '/', len(vocabulary)

    # Normalize each row (word vector) in the matrix to sum-up to 1
    row_norm = np.sum(np.abs(wv) ** 2, axis=-1) ** (1. / 2)
    wv /= row_norm[:, np.newaxis]

    word_index = {w: i for i, w in enumerate(words)}
    # pickle.dump((wv, word_index, word_set), open('{}.ebd_info'.format(file_name), 'wb'))
    return wv, word_index, word_set


def load_dataset(dataset_file, relations):
    """
    Loads a dataset file
    :param dataset_file: the file path
    :return: a list of dataset instances, (x, y, relation)
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        dataset = [tuple(line.strip().split('\t')) for line in f_in]
        dataset = {(x.lower(), y.lower()): relation for (x, y, relation) in dataset if relation in relations}

    return dataset


def unique(lst):
    """
    :param lst: a list of lists
    :return: a unique list of items appearing in those lists
    """
    indices = sorted(range(len(lst)), key=lst.__getitem__)
    indices = set(next(it) for k, it in
                  itertools.groupby(indices, key=lst.__getitem__))
    return [x for i, x in enumerate(lst) if i in indices]


def get_paths(corpus, x, y):
    """
    Get the paths that connect x and y in the corpus
    :param corpus: the corpus' resource object
    :param x: x
    :param y: y
    :return:
    """
    x_to_y_paths = corpus.get_relations(x, y)
    y_to_x_paths = corpus.get_relations(y, x)
    paths = {corpus.get_path_by_id(path): count for (path, count) in x_to_y_paths.iteritems()}
    paths.update({corpus.get_path_by_id(path).replace('X/', '@@@').replace('Y/', 'X/').replace('@@@', 'Y/'): count
                  for (path, count) in y_to_x_paths.iteritems()})
    return paths
