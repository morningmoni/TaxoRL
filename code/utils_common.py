import numpy as np
from collections import defaultdict
from itertools import count
import pickle
from tqdm import tqdm
import dynet as dy

from lstm_common import get_paths, vectorize_path, get_id
from features import get_all_features

from utils_tree import copy_tree
from __main__ import args


def load_paths_and_word_vectors(corpus, dataset_keys, lemma_index, keys, string_paths=None):
    # Define the dictionaries
    pos_index = defaultdict(count(0).next)
    dep_index = defaultdict(count(0).next)
    dir_index = defaultdict(count(0).next)

    _ = pos_index['#UNKNOWN#']
    _ = dep_index['#UNKNOWN#']
    _ = dir_index['#UNKNOWN#']

    # Vectorize tha paths
    # check for valid utf8 GB
    # keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    # keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in tqdm(dataset_keys)]
    print 'Get paths...'
    if string_paths is None:
        string_paths = [get_paths(corpus, x_id, y_id).items() for (x_id, y_id) in tqdm(keys)]
        if not args.debug:
            print 'saving string_paths...'
            pickle.dump(string_paths, open('pickled_data/string_paths_{}.pkl'.format(args.model_prefix_file), 'wb'))

    # Limit number of paths
    if args.max_paths_per_pair > 0:
        string_paths = [sorted(curr_paths, key=lambda x: x[1], reverse=True)[:args.max_paths_per_pair] for curr_paths in
                        string_paths]

    paths_x_to_y = [{vectorize_path(path, lemma_index, pos_index, dep_index, dir_index): count
                     for path, count in curr_paths}
                    for curr_paths in string_paths]
    paths = [{p: c for p, c in paths_x_to_y[i].iteritems() if p is not None} for i in range(len(keys))]

    # Get the word embeddings for x and y (get a lemma index)
    print 'Getting word vectors for the terms...'
    x_y_vectors = [(lemma_index.get(x, 0), lemma_index.get(y, 0)) for (x, y) in dataset_keys]

    print 'Getting features for x y...'
    hyper2hypo = pickle.load(open('../datasets/SemEval-2016/candidates_taxi/all_freq_twodatasets.pkl', 'rb'))
    hypo2hyper = defaultdict(lambda: defaultdict(int))
    for hyper in hyper2hypo:
        for hypo in hyper2hypo[hyper]:
            hypo2hyper[hypo][hyper] = hyper2hypo[hyper][hypo]
    lower2original = pickle.load(open('pickled_data/lower2original.pkl', 'rb'))
    features = [get_all_features(x, y, sub_feat=True, hyper2hypo=hyper2hypo, hypo2hyper=hypo2hyper,
                                 lower2original=lower2original) for (x, y) in tqdm(dataset_keys)]

    pos_inverted_index = {i: p for p, i in pos_index.iteritems()}
    dep_inverted_index = {i: p for p, i in dep_index.iteritems()}
    dir_inverted_index = {i: p for p, i in dir_index.iteritems()}

    dataset_instances = list(zip(paths, x_y_vectors, features))
    return dataset_instances, dict(pos_index), dict(dep_index), dict(dir_index), \
           pos_inverted_index, dep_inverted_index, dir_inverted_index


def check_error(name, v):
    if np.any(np.isnan(v.npvalue())) or np.any(np.isinf(v.npvalue())):
        print name, v.npvalue()
    else:
        print name, 'looks good [check_error]'


def check_error_np(name, v):
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        print name, v


def get_micro_f1(micro_total):
    if micro_total[0] == 0:
        return 0
    prec = micro_total[0] / micro_total[1]
    rec = micro_total[0] / micro_total[2]
    return round(2 * prec * rec / (prec + rec), 3)


def find_top_k(T_rollout, prob_per, pairs_per, k):
    prob_per = np.vstack(prob_per)
    pairs_per = np.vstack(pairs_per)
    # two-dim indices of those with higher prob
    indices_flat = np.argsort(prob_per.ravel())
    indices_pairs = indices_flat[-k:]
    indices = np.dstack(np.unravel_index(indices_flat, (len(prob_per), len(prob_per[0]))))[0][-k:]
    pair_from_tree_idx = [idx[0] for idx in indices]
    new_T_rollout = []
    idx_used = set()
    for idx in indices:
        # use original tree once
        if idx[0] not in idx_used:
            new_T_rollout.append(T_rollout[idx[0]])
            idx_used.add(idx[0])
        else:
            new_T_rollout.append(copy_tree(T_rollout[idx[0]], 1, nolist=True))
    prob_total = prob_per[indices[:, 0], indices[:, 1]]
    selected_pairs = pairs_per[indices_pairs]
    return prob_total, new_T_rollout, selected_pairs, pair_from_tree_idx


def test(epoch, trees_test, policy, policy_save_test, best_test, best_test_idx):
    if epoch % 10 == 0:
        pass
    metric_total = [0] * 6
    micro_total = [0.] * 3
    wrong_at_total = [0.] * 10
    policy.disable_dropout()
    for i_episode in tqdm(range(len(trees_test))):
        dy.renew_cg()
        policy.re_init()
        # prob_l = []
        T = trees_test[i_episode]
        T_rollout = copy_tree(T, min(args.n_rollout_test, (len(T.terms) - 1) * 2))  # a list of T's copy
        policy.init_history(args.n_rollout_test)

        for i in range(len(T.terms) - 1):
            if i == 0:
                prob, pairs = policy.selection_by_tree(T, mode='test')
                prob = dy.log(prob).npvalue()
                indices = np.argsort(prob)[-args.n_rollout_test:]
                prob_total = prob[indices]
                selected_pairs = [pairs[idx] for idx in indices]
                pair_from_tree_idx = [0] * len(T_rollout)
            else:
                prob_per = []
                pairs_per = []
                for T_idx in range(len(T_rollout)):
                    prob, pairs = policy.selection_by_tree(T_rollout[T_idx], mode='test', idx=T_idx)
                    prob = dy.log(prob) + prob_total[T_idx]
                    prob_per.append(prob.npvalue())
                    pairs_per.append(pairs)
                prob_total, T_rollout, selected_pairs, pair_from_tree_idx = find_top_k(T_rollout, prob_per, pairs_per,
                                                                                       args.n_rollout_test)
            for tree_idx, (tree_i, pair_i, from_idx) in enumerate(zip(T_rollout, selected_pairs, pair_from_tree_idx)):
                pair_i = tuple(pair_i)
                tree_i.update(pair_i)
                policy.update_history(pair_i, from_idx=from_idx, to_idx=tree_idx)
        # best candidate
        metric_total, micro_total, wrong_at_total, wrong_total = T.evaluate(metric_total, micro_total,
                                                                            wrong_at_total, reward_type='print')
        # if args.debug:
        #     for tmp_T in T_rollout:
        #         tmp_total = [0] * 6
        #         print tmp_T.evaluate(data=tmp_total, reward_type='print')
        # T.re_init()
    for m_idx in range(5):
        metric_total[m_idx] = round(metric_total[m_idx] / len(trees_test), 3)
    for w_idx in range(len(wrong_at_total)):
        wrong_at_total[w_idx] = round(wrong_at_total[w_idx] / len(trees_test), 3)
    best_test, best_test_idx = update_best(metric_total, best_test, best_test_idx, epoch)
    if epoch % 1 == 0:
        print '[test]epoch {}:{} {} {} {}'.format(epoch, metric_total, micro_total, get_micro_f1(micro_total),
                                                  wrong_at_total),
        print 'best_test', best_test, best_test_idx

    return policy_save_test, best_test, best_test_idx


def get_vocabulary(corpus, dataset_keys, path_lemmas=None):
    '''
    Get all the words in the dataset and paths
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :return: a set of distinct words appearing as x or y or in a path
    '''
    print '   word -> id ...'
    keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in tqdm(dataset_keys)]
    print '   path_lemmas ...'
    if path_lemmas is None:
        path_lemmas = set([edge.split('/')[0]
                           for (x_id, y_id) in tqdm(keys)
                           for path in get_paths(corpus, x_id, y_id).keys()
                           for edge in path.split('_')
                           if x_id > 0 and y_id > 0])
    print '   x_y_words ...'
    x_y_words = set([x for (x, y) in dataset_keys]).union([y for (x, y) in tqdm(dataset_keys)])
    return path_lemmas, x_y_words, keys
    # return list(path_lemmas.union(x_y_words))


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")


def sample_check(trees_test):
    sample_T = np.random.choice(trees_test)
    print sample_T.taxo
    print sample_T.taxo_test
    sample_T.evaluate(reward_type=args.reward, output=True)


def update_best(metric, best, best_i, epoch):
    for i in range(len(metric)):
        if metric[i] > best[i]:
            best[i] = metric[i]
            best_i[i] = epoch
    return best, best_i


def check_data(train_set, X_train, word_set):
    # how many samples have entity/path embeddings
    ebd_flag_train = []
    for i, (x, y) in enumerate(train_set.keys()):
        ebd_flag_train.append([])
        if x in word_set:
            ebd_flag_train[-1].append(1)
        if y in word_set:
            ebd_flag_train[-1].append(2)
        if len(X_train[i][0]) != 0:
            ebd_flag_train[-1].append(4)
    print("xy*={}, **p={}, x*p={}, *yp={}, xyp={} / {}".format(sum([sum(i) == 3 for i in ebd_flag_train]),
                                                               sum([sum(i) == 4 for i in ebd_flag_train]),
                                                               sum([sum(i) == 5 for i in ebd_flag_train]),
                                                               sum([sum(i) == 6 for i in ebd_flag_train]),
                                                               sum([sum(i) == 7 for i in ebd_flag_train]),
                                                               len(train_set)))


def check_limit(trees, policy, unk):
    scores_hyper = []
    scores_edge = []
    for T in trees:
        for hypo, hyper in T.taxo.items():
            if hyper == 'root007':
                continue
            if (hypo in unk or hyper in unk) and len(policy.get_data(hypo, hyper)[0]) == 0:
                continue
            T.update((hypo, hyper), test=True)
        scores_hyper.append(T.evaluate(reward_type='hyper'))
        scores_edge.append(T.evaluate(reward_type='edge'))
        T.re_init()
    print scores_hyper
    print np.mean(scores_hyper)
    print scores_edge
    print np.mean(scores_edge)


def save_path_info(dataset_keys, dataset_instances):
    # assert len(dataset_keys) == len(dataset_instances)
    pair2nPath = {}
    for k, v in zip(dataset_keys, dataset_instances):
        pair2nPath[k] = len(v[0])
    print 'saving num of paths with term-pairs as keys...'
    pickle.dump(pair2nPath, open('pair2nPath.pkl', 'wb'))
