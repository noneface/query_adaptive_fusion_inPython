# -*-coding:utf-8 -*-

import numpy as np
import load
from Eucdliean import dist
from normalize import normalize_curve
import matplotlib.pyplot as plt


def fusion():
    """fusion function to fuse each score curves"""

    # load image search score curve
    caffe_score = load.load_score_txt('score_caffe.txt')
    caffe_score = load.reshape_score(caffe_score, (1491, 500))
    print 'caffe score shape', caffe_score.shape

    gist_score = load.load_score_txt('score_gist.txt')
    gist_score = load.reshape_score(gist_score, (1491, 500))
    print 'gist score shape', gist_score.shape

    # load reference score curve
    ref_caffe_original = load.load_score_txt('reference_data/ref_caffe_100k.txt')
    x = ref_caffe_original.shape
    ref_caffe_original = load.reshape_score(ref_caffe_original, (x[0]/1490, 1490)).T
    print 'total caffe reference shape', ref_caffe_original.shape

    ref_gist_original = load.load_score_txt('reference_data/ref_gist_100k.txt')
    x = ref_gist_original.shape
    ref_gist_original = load.reshape_score(ref_gist_original, (x[0]/1490, 1490)).T
    print 'total gist reference shape', ref_gist_original.shape

    topN = 400  # parameter v in Eq. 4
    minN = 1  # parameter u in Eq. 4
    knn = 10  # number of selected references curves
    num_ref = 1000  # number of references to select from

    # reference selection from total of 100k references
    ref_caffe = ref_caffe_original[:, 0:num_ref].T
    print 'select caffe reference shape', ref_caffe.shape
    ref_gist = ref_gist_original[:, 0:num_ref].T
    print 'select bow reference shape', ref_gist.shape

    # total fusion features score
    score_final = None

    for n in range(500):
        #  image scores for the current query image
        score1 = caffe_score[:, n]
        score2 = gist_score[:, n]
        #  sort the scores
        score1_sorted = sorted(score1, reverse=True)
        print 'current caffe score length', len(score1_sorted)

        score2_sorted = sorted(score2, reverse=True)
        print 'current bow score length', len(score2_sorted)

        #  calculate the distance between the test score curve and references score curves.
        distance1 = dist(np.array(score1_sorted[minN:topN]).reshape(1, -1), ref_caffe[:, minN-1:topN-1])
        print 'caffe score and reference score distance shape', distance1.shape
        distance2 = dist(np.array(score2_sorted[minN:topN]).reshape(1, -1), ref_gist[:, minN-1:topN-1])
        print 'bow score and reference score distance shape', distance2.shape

        #  sort the distance
        i1 = np.argsort(distance1)
        i2 = np.argsort(distance2)

        #  select 1-knn reference curves in reference curves
        #  build a knn references matrix
        ref_caffe_selected = ref_caffe[i1[0, 0]]

        for i in i1[0, 1:knn]:
            ref_caffe_selected = np.vstack((ref_caffe_selected, ref_caffe[i]))
        print ref_caffe_selected.shape

        ref_gist_selected = ref_gist[i2[0, 0]]
        for i in i2[0, 1:knn]:
            ref_gist_selected = np.vstack((ref_gist_selected, ref_gist[i]))
        print ref_gist_selected.shape

        #  calculate the mean curves of the knn reference curves
        ref_caffe_final = np.mean(ref_caffe_selected, axis=0)
        print 'caffe reference score curve final', ref_caffe_final.shape

        ref_gist_final = np.mean(ref_gist_selected, axis=0)
        print 'bow reference score curve final', ref_gist_final.shape

        #  normalize the test score curves with calculated reference
        score1_norm = normalize_curve(np.array(score1_sorted[1:]), ref_caffe_final, topN)
        score2_norm = normalize_curve(np.array(score2_sorted[1:]), ref_gist_final, topN)

        #  calculate the area under the normalized score curve
        area1 = sum(score1_norm[minN:topN]) + 0.000000000001
        area2 = sum(score2_norm[minN:topN]) + 0.000000000001

        #  calculate the weights for each feature
        a = [(1 / area1) / (1 / area1 + 1 / area2), (1 / area2) / (1 / area1 + 1 / area2)]
        print 'weights', a

        #  use product rule to calculate final score
        score = score1**a[0] * score2**a[1]

        if score_final is None:
            score_final = score
        else:
            score_final = np.vstack((score_final, score))

        # plot curves for test
        plt.plot(range(100), score1_sorted[1:101], 'b-',
                 range(100), ref_caffe_final[:100], 'y-',
                 range(100), score1_norm[:100], 'r-', )

        plt.show()
        break
    return score_final

    # print score_final.shape

if __name__ == '__main__':
    fusion()
