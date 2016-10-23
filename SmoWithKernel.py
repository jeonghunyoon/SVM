# -*- coding: utf-8 -*-

import numpy as np
import random
import DataLoader

class Smo:
    def __init__(self, features, labels, c, tolerance, k_tup):
        self.X = np.matrix(features)
        self.y = np.matrix(labels).transpose()
        self.c = c
        self.tolerance = tolerance

        # 라그랑지 승수와 b 셋팅
        self.m, self.n = np.array(features).shape
        self.alphas = np.matrix(np.zeros((self.m, 1)))
        self.b = 0

        # error를 담기 위한 cache
        self.cache = np.matrix(np.zeros((self.m, 2)))

        # kernel trick을 위한 matrix
        self.K = np.matrix(np.zeros((self.m, self.m)))
        self.k_tup = k_tup
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], self.k_tup)

        # support vector를 구하기 위함. predict에서 사용한다.
        self.supp_vect_idx = []

    # calculate e_k
    def calc_e_k(self, k):
        fx_k = float(np.multiply(self.alphas, self.y).T * self.K[:, k] + self.b)
        e_k = fx_k - float(self.y[k])
        return e_k

    def select_j(self, i, e_i):
        max_k = -1
        max_delta_e = 0
        e_j = 0

        self.cache[i] = [1, e_i]
        valid_ecache_list = np.nonzero(self.cache[:, 0].A)[0]

        if (len(valid_ecache_list) > 1):
            for k in valid_ecache_list:
                if k==i:
                    continue
                e_k = self.calc_e_k(k)
                delta_e = abs(e_i - e_k)
                if (delta_e > max_delta_e):
                    max_k = k
                    max_delta_e = delta_e
                    e_j = e_k
            return max_k, e_j
        else:
            j = select_j_rand(i, self.m)
            e_j = self.calc_e_k(j)
        return j, e_j

    def update_e_k(self, k):
        e_k = self.calc_e_k(k)
        self.cache[k] = [1, e_k]

    def print_largrange(self):
        print(self.alphas, self.c, self.iter)

    def inner_l(self, i):
        # margin의 값
        e_i = self.calc_e_k(i)
        # 데이터와 초평면의 간격이 1보다 작아지게 될 때 (label = 1): e_i < 0, (label = -1): e_i > 0
        # 데이터와 초평면의 간격이 너무 멀어지게 될 때 (label = 1): e_i >> 0, (label = -1): e_i << 0
        if ((self.y[i] * e_i < -self.tolerance) and (self.alphas[i] < self.c)) or \
                ((self.y[i] * e_i > self.tolerance) and (self.alphas[i] > 0)):
            # random하게 j를 선택
            j, e_j = self.select_j(i, e_i)
            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()
            # alphas의 값이 0과 c사이에 있도록 한다.
            if (self.y[i] != self.y[j]):
                l = max(0, self.alphas[j] - self.alphas[i])
                h = min(self.c, self.c + self.alphas[j] - self.alphas[i])
            else:
                l = max(0, self.alphas[j] + self.alphas[i] - self.c)
                h = min(self.c, self.alphas[j] + self.alphas[i])

            if l == h:
                print "l==h"
                return 0

            # 2dot(x_i, x_j) - dot(x_i, x_i) - dot(x_j, x_j)
            eta = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j, j]

            if eta >= 0:
                print "eta>0"
                return 0

            # update alphas[j]
            self.alphas[j] -= self.y[j] * (e_i - e_j) / eta
            self.alphas[j] = clipAlpha(self.alphas[j], h, l)

            # j를 update
            self.update_e_k(j)

            if (abs(self.alphas[j] - alpha_j_old) < 0.00001):
                print "j not moving enough"
                return 0

            # update alphas[i]
            self.alphas[i] += self.y[j] * self.y[i] * (alpha_j_old - self.alphas[j])

            # i를 update
            self.update_e_k(i)

            b1 = self.b - e_i - self.y[i] * (self.alphas[i] - alpha_i_old) * self.K[i,i] - \
                self.y[j] * (self.alphas[j] - alpha_j_old) * self.K[i,j]

            b2 = self.b - e_j - self.y[i] * (self.alphas[i] - alpha_i_old) * self.K[i,j] - \
                self.y[j] * (self.alphas[j] - alpha_j_old) * self.K[j,j]

            if (0 < self.alphas[i]) and (self.c > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.c > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def run(self, max_interation):
        iteration = 0
        entire_set = True
        alpha_pairs_changed = 0

        while (iteration < max_interation) and ((alpha_pairs_changed > 0) or entire_set):
            alpha_pairs_changed = 0
            if entire_set:
                for i in range(self.m):
                    alpha_pairs_changed += self.inner_l(i)
                    print "full set, iter: %d, i: %d, pairs changed %d" %(iteration, i, alpha_pairs_changed)
                iteration += 1
            else:
                non_bound_ids = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.c))[0]
                for i in non_bound_ids:
                    alpha_pairs_changed += self.inner_l(i)
                    print "non-bound, iter: %d, i: %d, pairs changed %d" % (iteration, i, alpha_pairs_changed)
                iteration += 1
            if entire_set:
                entire_set = False
            elif (alpha_pairs_changed == 0):
                entire_set = True
            print "iteration number: %d" % iteration

            self.supp_vect_idx = np.nonzero(self.alphas.A > 0)[0]

        return self.b, self.alphas

    def calc_error_rate(self, test_set_loc):
        print "there are %d Support Vectors" % len(self.supp_vect_idx)
        error_count = 0

        for i in range(self.m):
            pred = self.predict(self.X[i, :])
            if np.sign(pred) != np.sign(self.y[i]):
                error_count += 1
        print "the training error rate is : %f" %(float(error_count) / self.m)

        data_loader = DataLoader.DataLoader()
        lab_test, feat_test = data_loader.load_data(test_set_loc)
        X_test = np.matrix(feat_test)
        y_test = np.matrix(lab_test).transpose()
        m_test, n_test = X_test.shape
        error_count = 0

        for i in range(m_test):
            pred = self.predict(X_test[i, :])
            if np.sign(pred) != np.sign(y_test[i]):
                error_count += 1
        print "the test error rate is : %f" % (float(error_count) / self.m)

    # predict 시, weight를 계산할 때, 전체데이터 셋보다는 support vector만 사용한다. alpha가 0인 경우를 제하기 위하여.
    def predict(self, x):
        kernel_eval = kernel_trans(self.X[self.supp_vect_idx], x, self.k_tup)
        pred = kernel_eval.T * np.multiply(self.y[self.supp_vect_idx], self.alphas[self.supp_vect_idx]) + self.b
        return pred

    def calc_w_s(self):
        w = np.zeros((self.n, 1))
        # KKT 조건을 통하여 w를 구하는 방법을 도출한다.
        for i in range(self.m):
            w += np.multiply(self.alphas[i] * self.y[i], self.X[i, :].T)
        return w

# SMO에서 사용할 i,j pair를 찾아준다. i가 고정되었을 때의 j를 구해줄 것이다.
def select_j_rand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# alpha j 의 값을 상한과 하한값의 사이에 둔다.
def clipAlpha(a_j, h, l):
    if a_j > h:
        a_j = h
    if a_j < l:
        a_j = l
    return a_j

# kernel을 계산
def kernel_trans(X, A, k_tup):
    m, n = np.shape(X)
    K = np.matrix(np.zeros((m, 1)))
    if k_tup[0] == 'lin':
        K = X * A.T
    elif k_tup[0] == 'rbf':
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = np.exp(K / (-1 * k_tup[1]**2))
    else:
        raise NameError('That kernel is not recognized')
    return K