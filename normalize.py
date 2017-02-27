# -*- coding:utf-8 -*-

from sklearn import preprocessing


def normalize_curve(score_sorted, ref, topN):
    score_norm = score_sorted - ref
    score_norm = (score_norm - min(score_norm[:topN]) + 0.000000000001) / \
                 (max(score_norm[:topN]) - min(score_norm[:topN]) + 0.000000000001)
    return score_norm


def normalized_l2(feature):
    return preprocessing.normalize(feature, norm='l2').T
