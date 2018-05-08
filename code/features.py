from difflib import SequenceMatcher
import numpy as np


def Capitalization(x, y, lower2original):
    if x in lower2original:
        x = lower2original[x]
    if y in lower2original:
        y = lower2original[y]
    if x[0].isupper() and y[0].isupper():
        return 0
    elif x[0].isupper() and not y[0].isupper():
        return 1
    elif not x[0].isupper() and y[0].isupper():
        return 2
    elif not x[0].isupper() and not y[0].isupper():
        return 3


def Endswith(x, y):
    return int(y.endswith(x))


def Contains(x, y):
    return int(x in y)


def Suffix_match(x, y):
    k = 7
    for i in range(k):
        if x[-i - 1:] != y[-i - 1:]:
            return i
    return k


def LCS(x, y):
    match = SequenceMatcher(None, x, y).find_longest_match(0, len(x), 0, len(y))
    res = 2.0 * match.size / (len(x) + len(y))  # [0, 1]
    return int(round(res, 1) * 10)  # [0,10]


def LD(x, y):
    res = 2.0 * (len(x) - len(y)) / (len(x) + len(y))  # (-2,2)
    return int(round(res, 1) * 10 + 20)  # [0, 40]


def normalized_freq_diff(hypo2hyper, x, y):
    if x not in hypo2hyper or y not in hypo2hyper[x] or hypo2hyper[x][y] == 0:
        a = 0
    else:
        a = float(hypo2hyper[x][y]) / max(hypo2hyper[x].values())
    if y not in hypo2hyper or x not in hypo2hyper[y] or hypo2hyper[y][x] == 0:
        b = 0
    else:
        b = float(hypo2hyper[y][x]) / max(hypo2hyper[y].values())
    res = a - b  # [-1, 1]
    # return res
    return int(res * 10) + 10  # [0, 20]


def generality_diff(hyper2hypo, x, y):
    if x not in hyper2hypo:
        b = 0
    else:
        b = np.log(1 + len([i for i in hyper2hypo[x] if hyper2hypo[x][i] != 0]))
    if y not in hyper2hypo:
        a = 0
    else:
        a = np.log(1 + len([i for i in hyper2hypo[y] if hyper2hypo[y][i] != 0]))
    res = a - b  # (-7.03, 7.02)
    # return res
    return int(res) + 7  # [0, 14]


def get_all_features(x, y, sub_feat=False, hypo2hyper=None, hyper2hypo=None, lower2original=None):
    feat = {}
    feat['Capitalization'] = Capitalization(x, y, lower2original)
    feat['Endswith'] = Endswith(x, y)
    feat['Contains'] = Contains(x, y)
    feat['Suffix_match'] = Suffix_match(x, y)
    feat['LCS'] = LCS(x, y)
    feat['LD'] = LD(x, y)
    if sub_feat:
        if hypo2hyper is None or hyper2hypo is None:
            print 'features.py: hypo2hyper not loaded'
            exit(-2)
        feat['Freq_diff'] = normalized_freq_diff(hypo2hyper, x, y)
        feat['General_diff'] = generality_diff(hyper2hypo, x, y)
    return feat
