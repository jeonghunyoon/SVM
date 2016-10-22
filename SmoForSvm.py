# -*- coding: utf-8 -*-

import numpy as np
import random


class Smo:
    def __init__(self, features, labels, c, tolerance):
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

    # calculate e_k
    def calc_e_k(self, k):
        fx_k = float(np.multiply(self.alphas, self.y).T * self.X * self.X[k, :].T) + self.b
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
        # print(np.shape(self.y))
        # print(np.shape(self.alphas))
        # print(np.shape(e_i))
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
            eta = 2.0 * self.X[i, :] * self.X[j, :].T - self.X[i, :] * self.X[i, :].T - self.X[j, :] * self.X[j, :].T

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

            b1 = self.b - e_i - self.y[i] * (self.alphas[i] - alpha_i_old) * \
                self.X[i, :] * self.X[i, :].T - \
                self.y[j] * (self.alphas[j] - alpha_j_old) * \
                self.X[i, :] * self.X[j, :].T

            b2 = self.b - e_j - self.y[i] * (self.alphas[i] - alpha_i_old) * \
                self.X[i, :] * self.X[j, :].T - \
                self.y[j] * (self.alphas[j] - alpha_j_old) * \
                self.X[j, :] * self.X[j, :].T

            if (0 < self.alphas[i]) and (self.c > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.c > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def run(self, max_interation, k_tup = ('lin', 0)):
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
        return self.b, self.alphas

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