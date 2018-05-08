import argparse
import os
import _dynet as dy
import pickle
from collections import defaultdict
from tqdm import tqdm

from model_RL import Policy
from utils_tree import read_tree_file, read_edge_files, load_candidate_from_pickle
from lstm_common import *
from evaluation_common import *
from knowledge_resource import KnowledgeResource
from features import *

ap = argparse.ArgumentParser()
# file path parameters
ap.add_argument('--corpus_prefix', default='../corpus/3in1_twodatasets/3in1_twodatasets',
                help='path to the corpus resource')
ap.add_argument('--dataset_prefix', default='../datasets/wn-bo', help='path to the train/test/val/rel data')
ap.add_argument('--model_prefix_file', default='twodatasets_subseqFeat', help='where to store the result')
ap.add_argument('--embeddings_file', default='../../Wikipedia_Word2vec/glove.6B.50d.txt',
                help='path to word embeddings file')
ap.add_argument('--trainname', default='train_wnbo_hyper', help='name of training data')
ap.add_argument('--valname', default='dev_wnbo_hyper', help='name of val data')
ap.add_argument('--testname', default='test_wnbo_hyper', help='name of test data')
# dimension parameters
ap.add_argument('--NUM_LAYERS', default=2, help='number of layers of LSTM')
ap.add_argument('--HIST_LSTM_HIDDEN_DIM', default=60)
ap.add_argument('--POS_DIM', default=4)
ap.add_argument('--DEP_DIM', default=5)
ap.add_argument('--DIR_DIM', default=1)
ap.add_argument('--MLP_HIDDEN_DIM', default=60)
ap.add_argument('--PATH_LSTM_HIDDEN_DIM', default=60)
# model settings
ap.add_argument('--max_paths_per_pair', type=int, default=200,
                help='limit the number of paths per pair. Invalid when loading from pkl')
ap.add_argument('--gamma', default=0.4)
ap.add_argument('--n_rollout', type=int, default=10, help='run for each sample')
ap.add_argument('--lr', default=1e-3, help='learning rate')
ap.add_argument('--choose_max', default=True, help='choose action with max prob when testing')
ap.add_argument('--allow_up', default=True, help='allow to attach some term as new root')
ap.add_argument('--reward', default='edge', choices=['hyper', 'edge', 'binary', 'fragment'])
ap.add_argument('--reward_form', default='diff', choices=['last', 'per', 'diff'])
# ablation parameters
ap.add_argument('--allow_partial', default=True, help='allow only partial tree is built')
ap.add_argument('--use_freq_features', default=True, help='use freq features')
ap.add_argument('--use_features', default=True, help='use surface features')
ap.add_argument('--use_path', default=True, help='use path-based info')
ap.add_argument('--use_xy_embeddings', default=True, help='use word embeddings')
# misc
ap.add_argument('--test_semeval', default=True, help='run tests on semeval datasets')
ap.add_argument('--load_model_file', default=None,
                help='if not None, load model from a file')
ap.add_argument('--load_opt', default=False, help='load opt along with the loaded model')
# parameters that are OUTDATED. may or may not affect performance
ap.add_argument('--word_dropout_rate', default=0.25, help='replace a token with <unk> with specified probability')
ap.add_argument('--path_dropout_rate', default=0, help='dropout of LSTM path embedding')
ap.add_argument('--no_training', default=False, help='load sample trees for training')
ap.add_argument('--debug', default=False, help='debug or normal run')
ap.add_argument('--n_rollout_test', type=int, default=5, help='beam search width')
ap.add_argument('--discard_rate', default=0., help='discard a pair w.o path info by discard_rate')
ap.add_argument('--set_max_height', default=False, help='limit the max height of tree')
ap.add_argument('--use_height_ebd', default=False, help='consider the height of each node')
ap.add_argument('--use_history', default=False, help='use history of taxonomy construction')
ap.add_argument('--use_sibling', default=False, help='use sibling signals')
ap.add_argument('--require_info', default=False, help='require there has to be info to infer...')
ap.add_argument('--given_root_train', default=False, help='[outdated]give gold root or not')
ap.add_argument('--given_root_test', default=False, help='[outdated]give gold root or not')
ap.add_argument('--filter_root', default=False, help='[outdated]filter root by term counts')
ap.add_argument('--one_layer', default=False, help='only one layer after pair representation')
ap.add_argument('--update_word_ebd', default=False, help='update word embedding or use fixed pre-train embedding')
ap.add_argument('--use_candidate', default=True, help='use candidates instead of considering all remaining pairs')
ap.add_argument('--height_ebd_dim', default=30)
args = ap.parse_args()

from utils_common import check_error, update_best, get_micro_f1, check_data, load_paths_and_word_vectors, \
    get_vocabulary, print_config, save_path_info, test, check_error_np

opt = vars(args)
score_filename = 'pickled_data/path{}_roll{}_debug{}.pkl'.format(args.max_paths_per_pair, args.n_rollout, args.debug)
n_run = 1
while os.path.exists(score_filename):
    score_filename = score_filename[:-len(str(n_run - 1))] + str(n_run)
    n_run += 1
print 'score_filename', score_filename
print('start time:{}'.format(time.ctime()))
print('last modified:{}'.format(time.ctime(os.path.getmtime(__file__))))


def main():
    print_config(opt)
    # Load the relations
    with codecs.open(args.dataset_prefix + '/relations.txt', 'r', 'utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = {relation: i for i, relation in enumerate(relations)}

    # Load the datasets
    if args.debug:
        trainname = '../datasets/wn-bo/train_sample.tsv'
        print 'Loading the dataset...', trainname, '*' * 10
        train_set = load_dataset(trainname, relations)
        val_set = load_dataset(trainname, relations)
        test_set = load_dataset(trainname, relations)
    else:
        trainname = '/' + args.trainname + '.tsv'
        valname = '/' + args.valname + '.tsv'
        testname = '/' + args.testname + '.tsv'
        print 'Loading the dataset...', trainname, '*' * 10
        train_set = load_dataset(args.dataset_prefix + trainname, relations)
        print 'Loading the dataset...', valname, '*' * 10
        val_set = load_dataset(args.dataset_prefix + valname, relations)
        print 'Loading the dataset...', testname, '*' * 10
        test_set = load_dataset(args.dataset_prefix + testname, relations)
    # y_train = [relation_index[label] for label in train_set.values()]
    # y_val = [relation_index[label] for label in val_set.values()]
    # y_test = [relation_index[label] for label in test_set.values()]
    dataset_keys = train_set.keys() + val_set.keys() + test_set.keys()
    # add (x, root) to dataset_keys
    vocab = set()
    for (x, y) in dataset_keys:
        vocab.add(x)
        vocab.add(y)
    dataset_keys += [(term, 'root007') for term in vocab]

    if not args.debug:
        trees = read_tree_file(
            "../datasets/wn-bo/wn-bo-trees-4-11-50-train533-lower.ptb",
            given_root=args.given_root_train, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_val = read_tree_file(
            "../datasets/wn-bo/wn-bo-trees-4-11-50-dev114-lower.ptb",
            given_root=args.given_root_test, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_test = read_tree_file(
            "../datasets/wn-bo/wn-bo-trees-4-11-50-test114-lower.ptb",
            given_root=args.given_root_test, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_semeval = read_edge_files("../datasets/SemEval-2016/original/",
                                        given_root=True, filter_root=args.filter_root, allow_up=False)
    else:
        trees = read_tree_file(
            "../datasets/wn-bo/train_sample.ptb2",
            given_root=args.given_root_train, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_val = read_tree_file(
            "../datasets/wn-bo/train_sample.ptb2",
            given_root=args.given_root_train, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_test = read_tree_file(
            "../datasets/wn-bo/train_sample.ptb2",
            given_root=args.given_root_test, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_semeval = read_tree_file(
            "../datasets/wn-bo/train_sample.ptb2",
            given_root=args.given_root_test, filter_root=args.filter_root, allow_up=args.allow_up)

    # Load the resource (processed corpus)
    print 'Loading the corpus...', args.corpus_prefix, '*' * 10
    corpus = KnowledgeResource(args.corpus_prefix)

    if not os.path.exists('pickled_data/preload_data_{}_debug{}.pkl'.format(args.model_prefix_file, args.debug)):
        print 'Loading the vocabulary...'
        # path_lemmas_name = "pickled_data/path_lemmas_3in1.pkl"
        # print 'reload path_lemmas from:', path_lemmas_name
        # path_lemmas = pickle.load(open(path_lemmas_name, 'rb'))
        path_lemmas, x_y_words, keys = get_vocabulary(corpus, dataset_keys, None)
        if not args.debug:
            pickle.dump(path_lemmas, open('pickled_data/path_lemmas_{}.pkl'.format(args.model_prefix_file), 'wb'))
            pickle.dump(x_y_words, open('pickled_data/x_y_words_{}.pkl'.format(args.model_prefix_file), 'wb'))

        # Load the word embeddings
        print 'Initializing word embeddings...'
        word_vectors, word_index, word_set = load_embeddings(args.embeddings_file, path_lemmas, x_y_words,
                                                             debug=args.debug)
        # Load the paths and create the feature vectors
        print 'Loading path files...'
        dataset_instances, pos_index, dep_index, dir_index, pos_inverted_index, dep_inverted_index, \
        dir_inverted_index = load_paths_and_word_vectors(corpus, dataset_keys, word_index, keys)
        print 'saving pkl...'
        pickle.dump((word_vectors, word_index, word_set, dataset_instances, pos_index, dep_index, dir_index,
                     pos_inverted_index, dep_inverted_index, dir_inverted_index),
                    open('pickled_data/preload_data_{}_debug{}.pkl'.format(args.model_prefix_file, args.debug), 'wb'))
    else:
        print 'Data loaded from', 'pickled_data/preload_data_{}_debug{}.pkl'.format(args.model_prefix_file,
                                                                                    args.debug), 'make sure pkl is correct'
        (word_vectors, word_index, word_set, dataset_instances, pos_index, dep_index, dir_index, pos_inverted_index,
         dep_inverted_index, dir_inverted_index) = pickle.load(
            open('pickled_data/preload_data_{}_debug{}.pkl'.format(args.model_prefix_file, args.debug), 'rb'))

    print 'Number of words %d, number of pos tags: %d, number of dependency labels: %d, number of directions: %d' % \
          (len(word_index), len(pos_index), len(dep_index), len(dir_index))

    # dataset_instances is now (paths, x_y_vectors, features)
    X_train = dataset_instances[:len(train_set)]
    X_val = dataset_instances[len(train_set):len(train_set) + len(val_set)]
    X_test = dataset_instances[len(train_set) + len(val_set):]
    print len(X_train), len(X_val), len(X_test)

    # check_data(train_set, X_train, word_set)
    # check_data(val_set, X_val, word_set)
    # check_data(test_set, X_test, word_set)
    # save_path_info(dataset_keys, dataset_instances)
    # scores_save = []
    # scores_save_test = []
    # prob_save = []
    # prob_save_test = []
    policy = Policy(dataset_keys, dataset_instances, num_lemmas=len(word_index), num_pos=len(pos_index),
                    num_dep=len(dep_index), num_directions=len(dir_index), opt=opt, num_relations=len(relations),
                    lemma_embeddings=word_vectors)
    trainer = dy.AdamTrainer(policy.model, alpha=args.lr)
    if args.debug:
        n_epoch = 1000
    else:
        n_epoch = 1000
    best = [0] * 6
    best_idx = [0] * 6
    best_val = [0] * 6
    best_val_idx = [0] * 6
    best_test = [0] * 6
    best_test_idx = [0] * 6
    best_semeval = [0] * 6
    best_semeval_idx = [0] * 6
    policy_save_test = defaultdict(list)
    wrong_total_l = []

    # check_limit(trees, policy, policy.unk_hard)
    # check_limit(trees, policy, policy.unk_soft)
    # check_limit(trees_test, policy, policy.unk_hard)
    # check_limit(trees_test, policy, policy.unk_soft)
    # exit(0)

    # TRAIN / TEST START HERE
    if args.load_model_file is None:
        for epoch in range(n_epoch):
            best, best_idx = train(epoch, trees, policy, trainer, best, best_idx, wrong_total_l)
            # policy_save_test, best_test, best_test_idx = test(epoch, trees_test, policy, policy_save_test, best_test,
            #                                                   best_test_idx)
            _, best_val, best_val_idx = test_single(epoch, trees_val, policy, [], best_val, best_val_idx, wrong_total_l)
            policy_save_test, best_test, best_test_idx = test_single(epoch, trees_test, policy, policy_save_test,
                                                                     best_test, best_test_idx, wrong_total_l)
    else:
        load_candidate_from_pickle(trees_semeval)
        _, best_semeval, best_semeval_idx = test_single(0, trees_semeval, policy, [], best_semeval,
                                                        best_semeval_idx, wrong_total_l,
                                                        reward_type='print_each')


def finish_episode(policy, trainer, entropy_l):
    loss = []
    all_cum_rewards = []
    for ct, p_rewards in enumerate(policy.rewards):
        R = 0
        rewards = []
        for r in p_rewards[::-1]:
            R = r + policy.gamma * R
            rewards.insert(0, R)
        all_cum_rewards.append(rewards)
        rewards = np.array(rewards) - policy.baseline_reward
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for action, reward, in zip(policy.saved_actions[ct], rewards):
            loss.append(-dy.log(action) * reward)
    # loss = dy.average(loss) + policy.decaying_beta * dy.average(entropy_l)
    loss = dy.average(loss)
    loss.backward()
    try:
        trainer.update()
        policy.update_baseline(np.mean(all_cum_rewards))
    except RuntimeError:
        print policy.rewards
        for actions in policy.saved_actions:
            for action in actions:
                print action.npvalue()
    policy.update_global_step()
    policy.update_eps()
    return loss.scalar_value()


def train(epoch, trees, policy, trainer, best, best_idx, wrong_total_l):
    # hyper edge fragment hyper-prec hyper-recall root
    metric_total = [0] * 6
    micro_total = [0.] * 3
    wrong_at_total = [0.] * 10
    np.random.shuffle(trees)
    loss = 0
    policy.set_dropout(args.path_dropout_rate)
    for i_episode in tqdm(range(len(trees))):
        T = trees[i_episode]
        entropy_l = []
        dy.renew_cg()
        policy.re_init()
        for _ in range(args.n_rollout):
            # prob_l = []
            policy.init_history()
            policy.rewards.append([])
            policy.saved_actions.append([])
            while len(T.V) > 0:
                pair, pr, entropy = select_action(T, policy, choose_max=False, mode='train')
                if pair is None:
                    break
                entropy_l.append(entropy)
                # prob_l.append(pr)
                T.update(pair)
                if args.reward_form != 'last' or len(T.V) == 0:
                    reward = T.eval(reward_type=args.reward, reward_form=args.reward_form)
                else:
                    reward = 0
                policy.rewards[-1].append(reward)
            metric_total, micro_total, wrong_at_total, wrong_total = T.evaluate(metric_total, micro_total,
                                                                                wrong_at_total, reward_type='print')
            wrong_total_l.append(wrong_total)
            # scores_save.append(T.evaluate(reward_type=REWARD, return_all=True))
            # prob_save.append(prob_l)
            T.re_init()
        loss += finish_episode(policy, trainer, entropy_l)
    for m_idx in range(5):
        metric_total[m_idx] = round(metric_total[m_idx] / len(trees) / args.n_rollout, 3)
    metric_total[0] = T.f1_calc(metric_total[3], metric_total[4])
    for w_idx in range(len(wrong_at_total)):
        wrong_at_total[w_idx] = round(wrong_at_total[w_idx] / len(trees) / args.n_rollout, 3)
    metric_total[5] /= args.n_rollout
    best, best_idx = update_best(metric_total, best, best_idx, epoch)
    if epoch % 1 == 0:
        print '[train]epoch {}:{} {} {} {}'.format(epoch, metric_total, micro_total, get_micro_f1(micro_total),
                                                   wrong_at_total),
        print 'total_loss', loss, 'best', best, best_idx
    return best, best_idx


def test_single(epoch, trees_test, policy, policy_save_test, best_test, best_test_idx, wrong_total_l,
                reward_type='print'):
    metric_total = [0] * 6
    micro_total = [0.] * 3
    wrong_at_total = [0.] * 10
    # if args.debug and epoch % 100 == 0:
    #     for T in trees_test:
    #         policy_save_test[T.rootname].append([])
    # elif not args.debug and epoch % 1 == 0:
    #     for T in trees_test:
    #         policy_save_test[T.rootname].append([])

    policy.disable_dropout()
    height_l = []
    for i_episode in range(len(trees_test)):
        dy.renew_cg()
        policy.re_init()
        # prob_l = []
        T = trees_test[i_episode]
        policy.init_history()
        policy.rewards.append([])
        policy.saved_actions.append([])
        if args.allow_up:
            n_time = len(T.terms) - 1
        else:
            n_time = len(T.terms)
        if reward_type == 'print_each':
            for _ in range(n_time):
                pair, pr, pairs, prob = select_action(T, policy, choose_max=args.choose_max, return_prob=True,
                                                      mode='test')
                if args.allow_partial and pair is None:
                    break
                T.update(pair)
                # print pair, pr,
                # T.evaluate(output=True)
        else:
            for _ in range(n_time):
                pair, pr, pairs, prob = select_action(T, policy, choose_max=args.choose_max, return_prob=True,
                                                      mode='test')
                if pair is None:
                    break
                T.update(pair)
                # print pair, pr,
                # T.evaluate(output=True)
            T.permute_ancestor()
        metric_total, micro_total, wrong_at_total, wrong_total = T.evaluate(metric_total, micro_total,
                                                                            wrong_at_total, reward_type=reward_type)
        wrong_total_l.append(wrong_total)
        height_l.append(T.cur_height)
        # T.draw()
        # T.save_for_vis(i_episode)
        # pickle.dump(T, open('{}.tree.pkl'.format(T.filename), 'wb'))`
        T.re_init()
        # scores_save_test.append(T.evaluate(reward_type=REWARD, return_all=True))
        # prob_save_test.append(prob_l)
    # sample_check(trees_test)
    for m_idx in range(5):
        metric_total[m_idx] = round(metric_total[m_idx] / len(trees_test), 3)
    metric_total[0] = T.f1_calc(metric_total[3], metric_total[4])
    for w_idx in range(len(wrong_at_total)):
        wrong_at_total[w_idx] = round(wrong_at_total[w_idx] / len(trees_test), 3)
    if metric_total[0] > 0.56 and args.load_model_file is None:
        policy.save_model('model_{}_epoch{}_{}'.format(args.model_prefix_file, epoch, metric_total[0]))
    best_test, best_test_idx = update_best(metric_total, best_test, best_test_idx, epoch)
    if epoch % 1 == 0:
        print '[test]epoch {}:{} {} {} {}'.format(epoch, metric_total, micro_total, get_micro_f1(micro_total),
                                                  wrong_at_total),
        print 'best_test', best_test, best_test_idx, np.mean(height_l), np.max(height_l), np.min(height_l)

    # pickle.dump((scores_save, scores_save_test, [], [], policy_save_test),
    #             open(score_filename, 'wb'))

    return policy_save_test, best_test, best_test_idx


def select_action(tree, policy, choose_max=False, return_prob=False, mode='train'):
    prob, pairs = policy.selection_by_tree(tree, mode)
    if pairs is None:
        if return_prob:
            return None, None, None, None
        else:
            return None, None, None
    with np.errstate(all='raise'):
        try:
            prob_v = prob.npvalue()
            if choose_max:
                idx = np.argmax(prob_v)
            else:
                # if np.random.random() < policy.epsilon:
                #     idx = np.random.randint(len(prob_v))
                #     while prob_v[idx] == 0:
                #         idx = np.random.randint(len(prob_v))
                # else:
                idx = np.random.choice(range(len(prob_v)), p=prob_v / np.sum(prob_v))
        except:
            for para in policy.model_parameters:
                check_error(para, dy.parameter(policy.model_parameters[para]))
            check_error('history', policy.history.output())
            check_error('pr', prob)
    action = prob[idx]
    policy.saved_actions[-1].append(action)
    policy.update_history(pairs[idx])
    if return_prob:
        return pairs[idx], prob_v[idx], pairs, prob_v
    return pairs[idx], prob_v[idx], dy.mean_elems(dy.cmult(prob, dy.log(prob)))


if __name__ == '__main__':
    main()
    print('end time:{}'.format(time.ctime()))
