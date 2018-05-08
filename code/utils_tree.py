import itertools
import re
from collections import defaultdict

import codecs
import numpy as np
import copy
import os
import pickle


def get_filtered_root(T, term_count, topk):
    cur_ct = [term_count[w] for w in T.V]
    cur_ct_idx = np.argsort(cur_ct)
    return set([T.V[idx] for idx in cur_ct_idx[-topk:]])


def isHyper(taxo, hypo, hyper):
    while hypo in taxo:
        if taxo[hypo] == hyper:
            return True
        hypo = taxo[hypo]
    return False


def isEdge(taxo, hypo, hyper):
    if hypo not in taxo:
        return False
    return taxo[hypo] == hyper


class node:
    def __init__(self, parent, name, level):
        self.parent = parent
        self.name = name
        self.children = []
        self.level = level


class Tree():
    def __init__(self, rootname, hypo2hyper, root_given, filter_root, term_count, allow_up, filename='',
                 term_conv=None):
        self.terms = set(hypo2hyper)
        self.term_conv = term_conv
        self.non_leaf_terms = set([v[0] for v in hypo2hyper.values()])
        if 'root007' in self.non_leaf_terms:
            self.non_leaf_terms.remove('root007')
        self.term2id = {t: ct for ct, t in enumerate(self.non_leaf_terms)}
        for term in self.terms - self.non_leaf_terms:
            self.term2id[term] = len(self.term2id)
        self.id2term = {v: k for k, v in self.term2id.items()}
        self.taxo = {}
        self.n_x2root_hyper = 0
        self.n_x2root_edge = 0
        self.rootname = rootname
        self.filename = filename
        self.last_pair = None
        self.root_given = root_given
        for k, v in hypo2hyper.items():
            self.taxo[k] = v[0]
            if v[0] == rootname:
                self.n_x2root_edge += 1
        self.hypo2hyper_set = set()
        gold_onehot = np.zeros((len(self.non_leaf_terms), len(self.terms)))
        for t1, t2 in itertools.permutations(self.terms, 2):
            if isHyper(self.taxo, t1, t2):
                self.hypo2hyper_set.add((t1, t2))
                gold_onehot[self.term2id[t2]][self.term2id[t1]] = 1
                if t2 == rootname:
                    self.n_x2root_hyper += 1
        self.gold_onehot = gold_onehot / np.linalg.norm(gold_onehot, axis=1)[:, np.newaxis]
        self.max_height = 10
        # V is the remaining vocab
        # N is terms on the tree
        self.allow_up = allow_up
        self.curroot_first = None
        self.hyper2hypo_candidate = defaultdict(set)
        self.re_init()
        if filter_root:
            self.filtered_root = get_filtered_root(self, term_count, 10)
        else:
            self.filtered_root = None

    def get_height(self, term):
        return self.term_height[term]

    def fragment_metric(self, display=False):
        tmp_term2id = {t: ct for ct, t in enumerate(self.N)}
        test_onehot = np.zeros((len(self.N), len(self.terms)))
        for hypo, hyper in self.hypo2hyper_test_set:
            test_onehot[tmp_term2id[hyper]][self.term2id[hypo]] = 1
        # remove 0s rows
        test_onehot = test_onehot[~np.all(test_onehot == 0, axis=1)]
        # normalize |v| = 1
        test_onehot = test_onehot / np.linalg.norm(test_onehot, axis=1)[:, np.newaxis]
        qd = np.dot(self.gold_onehot, test_onehot.T)
        tmp = np.argmax(qd, axis=1)
        qd = np.amax(qd, axis=1)
        if display:
            print self.hypo2hyper_test_set
            for t, i, score in zip(self.non_leaf_terms, tmp, qd):
                print [self.id2term[idx] for idx, v in enumerate(self.gold_onehot[self.term2id[t]]) if v > 0]
                print [self.id2term[idx] for idx, v in enumerate(test_onehot[i]) if v > 0]
                print score
        qd[qd < 0.2] = 0

        res = sum(qd) / len(self.non_leaf_terms)
        return res

    def re_init(self):
        self.taxo_test = {}
        self.hypo2hyper_test_set = set()
        self.hyper2hypo_edgeonly = defaultdict(list)
        self.hyper2hypo_candidate = defaultdict(set)
        # self.term_height = defaultdict(lambda: 1)
        self.term_height = defaultdict(int)
        self.V = list(self.terms)
        self.cur_height = 1
        if self.root_given:
            self.V.remove(self.rootname)
            self.N = [self.rootname]
            self.curroot = self.rootname
            self.term_height[self.curroot] = 1
        elif self.allow_up:
            # self.curroot = random.choice(self.V)
            # self.curroot_first = self.curroot
            if self.curroot_first is None:
                # self.curroot = random.choice(self.V)
                self.curroot = self.V[0]
                self.curroot_first = self.curroot
            else:
                self.curroot = self.curroot_first
            self.V.remove(self.curroot)
            self.N = [self.curroot]
            self.term_height[self.curroot] = 1
        else:
            self.N = []
        self.prev_eval = 0

    def save_best(self):
        self.taxo_test_best = {k: v for k, v in self.taxo_test.items()}
        self.hypo2hyper_test_set_best = set(self.hypo2hyper_test_set)
        self.root_best = self.N[0]

    def load_best(self):
        self.taxo_test = {k: v for k, v in self.taxo_test_best.items()}
        self.hypo2hyper_test_set = set(self.hypo2hyper_test_set_best)
        self.N = [self.root_best]

    def get_children(self, term):
        return self.hyper2hypo_edgeonly[term]

    def update(self, pair, test=False):
        self.last_pair = pair
        hypo = pair[0]
        hyper = pair[1]
        self.hyper2hypo_edgeonly[hyper].append(hypo)
        if not test:
            if not self.allow_up:
                self.N.append(hypo)
                try:
                    self.V.remove(hypo)
                except Exception as e:
                    print e
                    print hypo, hyper
                    exit(2)
            else:
                if hypo == self.curroot:
                    self.curroot = hyper
                    self.N.append(hyper)
                    self.V.remove(hyper)
                    self.term_height[hyper] = 1
                    for k in self.term_height:
                        self.term_height[k] += 1
                    self.cur_height += 1
                else:
                    self.N.append(hypo)
                    self.V.remove(hypo)
                    self.term_height[hypo] = self.term_height[hyper] + 1
                    self.cur_height = max(self.cur_height, self.term_height[hypo])

            assert len(self.V) + len(self.N) == len(self.terms)
            if not self.allow_up and hyper == 'root007':
                return
        self.taxo_test[hypo] = hyper
        cur = hypo
        while cur in self.taxo_test:
            self.hypo2hyper_test_set.add((hypo, self.taxo_test[cur]))
            cur = self.taxo_test[cur]
            # print 'pair selected:', pair, self.taxo_test, self.hypo2hyper_test_set

    def update_edgefile(self, pair):
        hypo = pair[0]
        hyper = pair[1]
        self.hyper2hypo_edgeonly[hyper].append(hypo)
        # if hypo in self.taxo_test:
        #     print '[warning]', hypo, 'already has a hypernym', self.taxo_test[hypo]
        self.taxo_test[hypo] = hyper

    def permute_ancestor_up2down(self):
        def _permute_ancestor_up2down(cur_hyper, real_hyper):
            for hypo in self.hyper2hypo_edgeonly[cur_hyper]:
                self.hypo2hyper_test_set.add((hypo, real_hyper))
                if hypo in self.hyper2hypo_edgeonly:
                    _permute_ancestor_up2down(hypo, real_hyper)

        for hyper in self.hyper2hypo_edgeonly:
            _permute_ancestor_up2down(hyper, hyper)

    def permute_ancestor(self):
        for hypo in self.taxo_test:
            cur = hypo
            while cur in self.taxo_test:
                self.hypo2hyper_test_set.add((hypo, self.taxo_test[cur]))
                cur = self.taxo_test[cur]

    def update_pairfile(self, pair):
        hypo = pair[0]
        hyper = pair[1]
        self.hyper2hypo_edgeonly[hyper].append(hypo)
        self.taxo_test[hypo] = hyper
        cur = hypo
        self.hypo2hyper_test_set.add((hypo, self.taxo_test[cur]))

    def draw(self):
        q = [self.curroot]
        nextq = []
        print q
        while len(q) > 0:
            cur = q.pop(0)
            print self.hyper2hypo_edgeonly[cur],
            nextq.extend(self.hyper2hypo_edgeonly[cur])
            if len(q) == 0:
                q = nextq
                nextq = []
                print

    def eval(self, reward_type, reward_form):
        cur_reward = self.evaluate(reward_type=reward_type)
        if reward_form == 'per' or reward_form == 'last':
            return cur_reward
        elif reward_form == 'diff':
            reward = cur_reward - self.prev_eval
            self.prev_eval = cur_reward
            return reward
        else:
            print "no such reward form:", reward_form
            raise NotImplementedError

    def wrong_at(self, n_wrong, k):
        if n_wrong <= k:
            return 1
        else:
            return 0

    def evaluate(self, data=None, micro_data=None, wrong_at_data=None, reward_type='', output=False, return_all=False,
                 display=False):
        res_isHyper_Test = [i in self.hypo2hyper_set for i in self.hypo2hyper_test_set]
        res_isEdge_Test = [self.taxo[k] == v for k, v in self.taxo_test.items()]
        if len(res_isHyper_Test) == 0:
            if len(self.N) == 0:
                res_isHyper_Test = [False]
                res_isEdge_Test = [False]
            else:
                if self.N[0] == self.rootname:
                    res_isHyper_Test = [True]
                    res_isEdge_Test = [True]
                else:
                    res_isHyper_Test = [False]
                    res_isEdge_Test = [False]

        n_hyper = sum(res_isHyper_Test)
        n_edge = sum(res_isEdge_Test)
        hyper_prec = n_hyper / float(len(res_isHyper_Test))
        hyper_recall = n_hyper / float(len(self.hypo2hyper_set))
        if hyper_prec == 0 and hyper_recall == 0:
            hyper_f1 = 0
        else:
            hyper_f1 = 2 * hyper_recall * hyper_prec / (hyper_recall + hyper_prec)
        edge_recall = n_edge / float(len(self.taxo) - 1)  # realroot -> 'root'
        edge_prec = n_edge / float(len(res_isEdge_Test))
        if edge_prec == 0 and edge_recall == 0:
            edge_f1 = 0
        else:
            edge_f1 = 2 * edge_recall * edge_prec / (edge_recall + edge_prec)
        if output or reward_type == 'print_each':
            # print '#pairs (x, root):{} ** #edges (x, root):{}'.format(self.n_x2root_hyper, self.n_x2root_edge)
            print "Hyper-Prec: {} / {} = {:.3f}".format(sum(res_isHyper_Test), len(res_isHyper_Test), hyper_prec),
            print "Hyper-recall: {} / {} = {:.3f}".format(sum(res_isHyper_Test), len(self.hypo2hyper_set),
                                                          hyper_recall),
            print "Hyper-F1: {:.3f}".format(hyper_f1),
            print "Edge-Prec: {} / {} = {:.3f}".format(sum(res_isEdge_Test), len(self.taxo_test), edge_prec),
            print "Edge-recall = {} / {} = {:.3f}".format(sum(res_isEdge_Test), len(self.taxo) - 1, edge_recall),
            print "Edge-F1 = {:.3f}".format(edge_f1)
            if reward_type != 'print_each':
                return
        if 'print' in reward_type:
            cur_data = [0.] * len(data)
            if not self.allow_up and self.N[0] == self.rootname:
                cur_data[5] = 1.
            elif self.allow_up and self.curroot == self.rootname:
                cur_data[5] = 1.
            cur_data[0] = hyper_f1
            cur_data[1] = edge_f1
            cur_data[2] = self.fragment_metric(display)
            cur_data[3] = hyper_prec
            cur_data[4] = hyper_recall
            for i in range(len(data)):
                data[i] += cur_data[i]
            if reward_type == 'print_each':
                print self.filename, cur_data

            micro_data[0] += n_hyper
            micro_data[1] += len(res_isHyper_Test)
            micro_data[2] += len(self.hypo2hyper_set)

            wrong_idx = 0
            for k in [5, 10, 20, 30, 40]:
                wrong_at_data[wrong_idx] += self.wrong_at(len(res_isHyper_Test) - n_hyper, k)
                wrong_idx += 1
                wrong_at_data[wrong_idx] += self.wrong_at(len(res_isEdge_Test) - n_edge, k)
                wrong_idx += 1
            return data, micro_data, wrong_at_data, (len(res_isHyper_Test) - n_hyper, len(res_isEdge_Test) - n_edge)
        if return_all:
            return hyper_prec, hyper_recall, hyper_f1, edge_prec, edge_recall, edge_f1, self.n_x2root_hyper, self.n_x2root_edge
        elif reward_type == 'fragment':
            return self.fragment_metric()
        if reward_type == 'edge':
            # if len(self.V) == 0:
            #     return edge_f1 * 3
            return edge_f1
        elif reward_type == 'edge-prec':
            return edge_prec
        elif reward_type == 'edge-recall':
            return edge_recall
        elif reward_type == 'hyper':
            return hyper_f1
        elif reward_type == 'hyper-prec':
            return hyper_prec
        elif reward_type == 'hyper-recall':
            return hyper_recall
        elif reward_type == 'binary':
            return int(self.taxo[self.last_pair[0]] == self.last_pair[1])

        print 'no such reward_type:', reward_type
        raise NotImplementedError

    @staticmethod
    def f1_calc(p, r):
        return round(2 * p * r / (p + r), 3)


def copy_tree(tree, times, nolist=False):
    if nolist:
        return copy.deepcopy(tree)

    return [copy.deepcopy(tree) for _ in range(times)]


def read_tree(line):
    level = 0
    root = node(None, 'root007', level)
    level += 1
    curnode = root
    term = ''
    level_down, level_up = 0, 0
    for c in line:
        if c == '(':
            level_down += 1
        elif c == ' ' or c == '\n':
            curnode.children.append(node(curnode, term, level))
            while level_down != 0:
                level += 1
                curnode = curnode.children[-1]
                level_down -= 1
            while level_up != 0:
                level -= 1
                curnode = curnode.parent
                level_up -= 1
            term = ''
        elif c == ')':
            level_up += 1
        else:
            term += c
    return root


def traverse(root, h):
    #     print(root.level, root.name, '->', [i.name for i in root.children])
    for i in root.children:
        h[re.sub('_\$_', '_', i.name)].append(re.sub('_\$_', '_', root.name))
    for i in root.children:
        traverse(i, h)


def read_tree_file(in_name, given_root, filter_root=False, allow_up=True, noUnderscore=False):
    trees = []
    # term_count = get_termcount()
    term_count = None
    with open(in_name) as f:
        # each line is a tree
        for line in f:
            root = read_tree(line)
            realroot = re.sub('_\$_', '_', root.children[0].name)
            hypo2hyper_edgeonly = defaultdict(list)
            # after traverse, edges of a tree are added to hypo2hyper
            traverse(root, hypo2hyper_edgeonly)
            if noUnderscore:
                hypo2hyper_edgeonly_noUnderscore = defaultdict(list)
                for k, v in hypo2hyper_edgeonly.items():
                    hypo2hyper_edgeonly_noUnderscore[re.sub('_', '', k)].append(re.sub('_', '', v[0]))
                hypo2hyper_edgeonly = hypo2hyper_edgeonly_noUnderscore
            trees.append(
                Tree(realroot, hypo2hyper_edgeonly, root_given=given_root, filter_root=filter_root,
                     term_count=term_count, allow_up=allow_up))
    return trees


def read_edge_files(in_path, given_root=False, filter_root=False, allow_up=True, noUnderscore=False):
    trees = []
    for root, dirs, files in os.walk(in_path):
        for filename in files:
            if not filename.endswith('taxo'):
                continue
            file_path = root + filename
            print 'read_edge_files', file_path
            with codecs.open(file_path, 'r', 'utf-8') as f:
                hypo2hyper_edgeonly = defaultdict(list)
                terms = set()
                noSpace2underscore = {}
                for line in f:
                    hypo, hyper = line.strip().lower().split('\t')[1:]
                    hypo_noSpace = re.sub(' ', '', hypo)
                    hyper_noSpace = re.sub(' ', '', hyper)
                    hypo_ = re.sub(' ', '_', hypo)
                    hyper_ = re.sub(' ', '_', hyper)
                    noSpace2underscore[hypo_noSpace] = hypo_
                    noSpace2underscore[hyper_noSpace] = hyper_
                    if noUnderscore:
                        terms.add(hypo_noSpace)
                        terms.add(hyper_noSpace)
                        hypo2hyper_edgeonly[hypo_noSpace].append(hyper_noSpace)
                    else:
                        terms.add(hypo_)
                        terms.add(hyper_)
                        hypo2hyper_edgeonly[hypo_].append(hyper_)
                realroot = list(terms - set(hypo2hyper_edgeonly))[0]
                hypo2hyper_edgeonly[realroot].append('root007')
                trees.append(
                    Tree(realroot, hypo2hyper_edgeonly, root_given=given_root, filter_root=filter_root,
                         term_count=None, allow_up=allow_up, filename=filename, term_conv=noSpace2underscore))
    return trees


def load_candidate_from_pickle(trees):
    for T in trees:
        ct = 0
        ct_substr = 0
        hyper2hypo_w_freq = pickle.load(
            open('../datasets/SemEval-2016/candidates_taxi/{}.pkl'.format(T.filename + '.candidate_w_freq'), 'rb'))
        for hyper in hyper2hypo_w_freq:
            for hypo in hyper2hypo_w_freq[hyper]:
                if hyper2hypo_w_freq[hyper][hypo] >= 20:
                    ct += 1
                    T.hyper2hypo_candidate[hyper].add(hypo)
        for hypo, hyper in itertools.permutations(T.terms, 2):
            if hyper in hypo:
                ct_substr += 1
                T.hyper2hypo_candidate[hyper].add(hypo)
        print 'load {}+{}={} candidates for tree {} w. size {}'.format(ct, ct_substr,
                                                                       ct + ct_substr,
                                                                       T.filename,
                                                                       len(T.terms))
