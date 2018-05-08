import pickle
import _dynet as dy
import numpy as np
import json

feat_dims = {'Endswith': (2, 10), 'Contains': (2, 10), 'Suffix_match': (8, 10),
             'LCS': (11, 10), 'LD': (41, 10), 'Freq_diff': (21, 10), 'General_diff': (15, 10)}
# feat_dims['Capitalization'] = (4, 10)
EMPTY_PATH = ((0, 0, 0, 0),)

dyparams = dy.DynetParams()
dyparams.set_random_seed(478865112)
dyparams.set_cpu_mode()
np.random.seed(1024)
dyparams.init()


class Policy:
    def __init__(self, dataset_keys, data, num_lemmas, num_pos, num_dep, opt,
                 num_directions=5, num_relations=2, lemma_embeddings=None):
        self.dataset_keys = {k: i for i, k in enumerate(dataset_keys)}
        self.data = data
        self.num_lemmas = num_lemmas
        self.num_pos = num_pos
        self.num_dep = num_dep
        self.num_directions = num_directions
        self.num_relations = num_relations
        self.lemma_vectors = None
        if lemma_embeddings is not None:
            self.lemma_vectors = lemma_embeddings
            self.lemma_embeddings_dim = lemma_embeddings.shape[1]
        else:
            self.lemma_embeddings_dim = 50
        self.opt = opt
        self.f_cache = {}
        self.path_cache = {}
        self.saved_actions = []
        self.rewards = []
        self.baseline_reward = 0
        self.gamma = self.opt['gamma']
        self.L = 0.02
        self.decaying_beta = 0.05
        self.epsilon = 1
        self.decaying_factor = 0.9
        self.global_step = 0
        if self.opt['require_info']:
            self.unk_hard = pickle.load(open('pickled_data/unk_hard.pkl', 'rb'))
            self.unk_soft = pickle.load(open('pickled_data/unk_soft.pkl', 'rb'))

        print 'Creating the network...'
        self.builder, self.model, self.model_parameters, self.builder_hist, self.f_dim = self.create_computation_graph(
            self.num_lemmas, self.num_pos,
            self.num_dep,
            self.num_directions,
            self.num_relations,
            self.lemma_vectors,
            self.lemma_embeddings_dim)
        self.history = None
        print 'pair embedding dim:', self.f_dim

    def save_model(self, filename):
        print 'saving model {}...'.format(filename)
        self.model.save('{}.model'.format(filename))
        with open('{}.json'.format(filename), 'w') as fp:
            json.dump(self.opt, fp)

    def re_init(self):
        del self.rewards[:]
        del self.saved_actions[:]
        self.f_cache.clear()
        self.path_cache.clear()

    def update_baseline(self, target):
        self.baseline_reward = self.L * target + (1 - self.L) * self.baseline_reward

    def update_global_step(self):
        self.global_step += 1

    def update_beta(self):
        if self.global_step % 533 == 0:
            self.decaying_beta *= self.decaying_factor

    def update_eps(self):
        if self.global_step % 533 == 0 and self.epsilon > .1:
            self.epsilon *= .9
            # print self.epsilon, 'epsilon'

    def process_one(self, i, j, tree, mode):
        def process_one_instance(instance, update=True, x_y_vectors=None, features=None, mode='train'):
            lemma_lookup = self.model_parameters['lemma_lookup']
            if self.opt['use_path']:
                pos_lookup = self.model_parameters['pos_lookup']
                dep_lookup = self.model_parameters['dep_lookup']
                dir_lookup = self.model_parameters['dir_lookup']
                # Add the empty path
                paths = instance
                if len(paths) == 0:
                    paths[EMPTY_PATH] = 1

                # Compute the averaged path
                num_paths = reduce(lambda x, y: x + y, instance.itervalues())
                path_embeddings = [
                    self.get_path_embedding_from_cache(lemma_lookup, pos_lookup, dep_lookup, dir_lookup, path, update,
                                                       mode) * count
                    for path, count in instance.iteritems()]
                input_vec = dy.esum(path_embeddings) * (1.0 / num_paths)

            # Concatenate x and y embeddings
            if self.opt['use_xy_embeddings']:
                x_vector, y_vector = dy.lookup(lemma_lookup, x_y_vectors[0]), dy.lookup(lemma_lookup, x_y_vectors[1])
                if self.opt['use_path']:
                    input_vec = dy.concatenate([x_vector, input_vec, y_vector])
                else:
                    input_vec = dy.concatenate([x_vector, y_vector])
            if self.opt['use_features']:
                for k in feat_dims:
                    if 'diff' in k and not self.opt['use_freq_features']:
                        continue
                    feat = dy.lookup(self.model_parameters[k], features[k])
                    input_vec = dy.concatenate([input_vec, feat])

            if self.opt['use_height_ebd']:
                if j in tree.term_height:
                    h = tree.get_height(j) - 1
                else:
                    h = 0
                height_vector = dy.lookup(self.model_parameters['height_lookup'], h)
                input_vec = dy.concatenate([input_vec, height_vector])
            return input_vec

        if (i, j) not in self.f_cache:
            data = self.get_data(i, j)
            f = process_one_instance(instance=data[0], update=self.opt['update_word_ebd'], x_y_vectors=data[1],
                                     features=data[2], mode=mode)
            self.f_cache[(i, j)] = f
        if not self.opt['use_sibling']:
            # return dy.concatenate([self.f_cache[(i, j)], self.history[0].output()])
            return self.f_cache[(i, j)]
        else:
            sib = [self.f_cache[(sibling, j)] for sibling in tree.get_children(j)]
            if len(sib) == 0:
                return self.f_cache[(i, j)]
            else:
                return self.f_cache[(i, j)] + dy.esum(sib) / len(sib)

    def get_data(self, i, j):
        '''
        :param i: term1
        :param j: term2
        :return: (instance, x_y_vectors, features)
        '''
        idx = self.dataset_keys[(i, j)]
        return self.data[idx]

    def _select_by_tree(self, tree, mode, discard):
        input_layers = []
        pairs = []
        for i in tree.V:
            if len(tree.N) != 0:
                for j in tree.N:
                    if discard and tree.filename != '' and self.opt['use_candidate']:
                        if i not in tree.hyper2hypo_candidate[j]:
                            continue
                    # if self.opt['set_max_height'] and tree.get_height(j) >= tree.max_height:
                    #     continue
                    # if discard and self.opt['require_info']:
                    #     if i in self.unk_hard:
                    #         continue
                    #     if len(self.get_data(i, j)[0]) == 0 and random.random() < self.opt['discard_rate']:
                    #         continue
                    input_layers.append(self.process_one(i, j, tree, mode))
                    pairs.append((i, j))
                if tree.allow_up:
                    if tree.filename != '' and self.opt['use_candidate']:
                        if tree.curroot not in tree.hyper2hypo_candidate[i]:
                            continue
                    # if self.opt['set_max_height'] and tree.cur_height >= tree.max_height:
                    #     continue
                    # unk can't be root -> no discard flag
                    # if self.opt['require_info']:
                    #     if i in self.unk_hard:
                    #         continue
                    #     if len(self.get_data(tree.curroot, i)[0]) == 0 and random.random() < self.opt['discard_rate']:
                    #         continue
                    input_layers.append(self.process_one(tree.curroot, i, tree, mode))
                    pairs.append((tree.curroot, i))
            # if root not given, use virtual root
            else:
                # if tree.filtered_root and i not in tree.filtered_root:
                #     continue
                input_layers.append(self.process_one(i, 'root007', tree, mode))
                pairs.append((i, 'root007'))
        return input_layers, pairs

    def selection_by_tree(self, tree, mode, idx=0):
        input_layers, pairs = self._select_by_tree(tree, mode, True)
        if len(pairs) == 0:
            if not self.opt['allow_partial']:
                input_layers, pairs = self._select_by_tree(tree, mode, False)
            else:
                print 'early stop! discard {} / {}.'.format(len(tree.V), len(tree.terms))
                return None, None
        W1_rl = dy.parameter(self.model_parameters['W1_rl'])
        b1_rl = dy.parameter(self.model_parameters['b1_rl'])
        if not self.opt['one_layer']:
            W2_rl = dy.parameter(self.model_parameters['W2_rl'])
            b2_rl = dy.parameter(self.model_parameters['b2_rl'])

        # pr = W2_rl * dy.rectify(W1_rl * dy.concatenate_to_batch(input_layers) + b1_rl) + b2_rl
        # (V x N)x160 160x50 50x60 60x1
        input_layers = dy.concatenate_cols(input_layers)
        input_layers = dy.transpose(input_layers)

        if not self.opt['one_layer']:
            if self.opt['use_history']:
                pr = input_layers * dy.rectify(W2_rl * dy.rectify(W1_rl * self.history[idx].output() + b1_rl) + b2_rl)
            else:
                pr = dy.rectify(input_layers * W2_rl + b2_rl) * W1_rl + b1_rl
        else:
            if self.opt['use_history']:
                pr = input_layers * dy.rectify(W1_rl * self.history[idx].output() + b1_rl)
            else:
                pr = input_layers * W1_rl + b1_rl
        # (#actions, )
        pr = dy.reshape(pr, (len(pairs),))
        return dy.softmax(pr), pairs

    def init_history(self, num=1):
        x1 = dy.vecInput(self.f_dim)
        self.history = [self.builder_hist.initial_state().add_input(x1) for _ in range(num)]

    def update_history(self, pair, from_idx=0, to_idx=0):
        self.history[to_idx] = self.history[from_idx].add_input(self.f_cache[pair])

    def get_path_embedding_from_cache(self, lemma_lookup, pos_lookup, dep_lookup, dir_lookup, path, update, mode):
        if path not in self.path_cache:
            self.path_cache[path] = self.get_path_embedding(lemma_lookup, pos_lookup, dep_lookup,
                                                            dir_lookup, path, update, mode)
        return self.path_cache[path]

    def get_path_embedding(self, lemma_lookup, pos_lookup, dep_lookup, dir_lookup, path, update=True, mode='train'):
        # Concatenate the edge components to one vector
        inputs = [dy.concatenate([self.word_dropout(lemma_lookup, edge[0], update, mode=mode),
                                  self.word_dropout(pos_lookup, edge[1], mode=mode),
                                  self.word_dropout(dep_lookup, edge[2], mode=mode),
                                  self.word_dropout(dir_lookup, edge[3], mode=mode)])
                  for edge in path]

        return self.builder.initial_state().transduce(inputs)[-1]

    def word_dropout(self, lookup_table, word, update=True, mode='train'):
        if mode == 'train':
            if word != 0 and np.random.random() < self.opt['word_dropout_rate']:
                word = 0
        return dy.lookup(lookup_table, word, update)

    def create_computation_graph(self, num_lemmas, num_pos, num_dep, num_directions, num_relations,
                                 wv=None, lemma_dimension=50):
        model = dy.ParameterCollection()
        if self.opt['use_path']:
            input_dim = self.opt['PATH_LSTM_HIDDEN_DIM']
        else:
            input_dim = 0

        # dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
        builder = dy.LSTMBuilder(self.opt['NUM_LAYERS'],
                                 lemma_dimension + self.opt['POS_DIM'] + self.opt['DEP_DIM'] + self.opt['DIR_DIM'],
                                 input_dim, model)

        model_parameters = {}

        for k, v in feat_dims.items():
            model_parameters[k] = model.add_lookup_parameters(v)
        # Concatenate x and y
        if self.opt['use_xy_embeddings']:
            input_dim += 2 * lemma_dimension
        if self.opt['use_features']:
            for name, dim in feat_dims.items():
                if 'diff' in name and not self.opt['use_freq_features']:
                    continue
                input_dim += dim[1]
        if self.opt['use_height_ebd']:
            model_parameters['height_lookup'] = model.add_lookup_parameters((10, self.opt['height_ebd_dim']))
            input_dim += self.opt['height_ebd_dim']

        model_parameters['lemma_lookup'] = model.add_lookup_parameters((num_lemmas, lemma_dimension))
        builder_hist = dy.LSTMBuilder(2, input_dim, self.opt['HIST_LSTM_HIDDEN_DIM'], model)

        # Pre-trained word embeddings
        if wv is not None:
            model_parameters['lemma_lookup'].init_from_array(wv)

        model_parameters['pos_lookup'] = model.add_lookup_parameters((num_pos, self.opt['POS_DIM']))
        model_parameters['dep_lookup'] = model.add_lookup_parameters((num_dep, self.opt['DEP_DIM']))
        model_parameters['dir_lookup'] = model.add_lookup_parameters((num_directions, self.opt['DIR_DIM']))

        if not self.opt['one_layer']:
            if self.opt['use_history']:
                model_parameters['W2_rl'] = model.add_parameters((input_dim, self.opt['MLP_HIDDEN_DIM']))
                model_parameters['b2_rl'] = model.add_parameters((input_dim, 1))
                model_parameters['W1_rl'] = model.add_parameters(
                    (self.opt['MLP_HIDDEN_DIM'], self.opt['HIST_LSTM_HIDDEN_DIM']))
                model_parameters['b1_rl'] = model.add_parameters((self.opt['MLP_HIDDEN_DIM'], 1))
            else:
                model_parameters['W2_rl'] = model.add_parameters((input_dim, self.opt['MLP_HIDDEN_DIM']))
                model_parameters['b2_rl'] = model.add_parameters((1, self.opt['MLP_HIDDEN_DIM']))
                model_parameters['W1_rl'] = model.add_parameters((self.opt['MLP_HIDDEN_DIM'], 1))
                model_parameters['b1_rl'] = model.add_parameters((1, 1))
        else:
            if self.opt['use_history']:
                model_parameters['W1_rl'] = model.add_parameters((input_dim, self.opt['HIST_LSTM_HIDDEN_DIM']))
                model_parameters['b1_rl'] = model.add_parameters((input_dim, 1))
            else:
                model_parameters['W1_rl'] = model.add_parameters((input_dim, 1))
                model_parameters['b1_rl'] = model.add_parameters((1, 1))

        if self.opt['load_model_file'] is not None:
            print 'model loaded from', self.opt['load_model_file']
            model.populate('{}'.format(self.opt['load_model_file']))
            if self.opt['load_opt']:
                print 'opt loaded from', '{}.json'.format(self.opt['load_model_file'])
                self.opt = json.load(open('{}.json'.format(self.opt['load_model_file'])))

        return builder, model, model_parameters, builder_hist, input_dim

    def set_dropout(self, p):
        self.builder.set_dropout(p)

    def disable_dropout(self):
        self.builder.disable_dropout()
