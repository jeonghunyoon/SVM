# -*- coding: utf-8 -*-

import numpy as np
import random


class SmoSimple:
    def __init__(self, features, labels, c, tolerance, max_iterations):
        self.features = np.matrix(features)
        self.labels = np.matrix(labels).transpose()
        self.c = c
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # 라그랑지 승수와 b 셋팅
        self.m, self.n = np.array(features).shape
        self.alphas = np.matrix(np.zeros((self.m, 1)))
        self.b = 0

        self.iter = 0

    def print_largrange(self):
        print(self.alphas, self.c, self.iter)

    def run(self):
        while (self.iter < self.max_iterations):
            alpha_pairs_changed = 0
            for i in range(self.m):
                # model의 결과값
                fx_i = float(np.multiply(self.alphas, self.labels).T * self.features * self.features[i, :].T + self.b)
                # margin의 값
                e_i = fx_i - float(self.labels[i])
                # 데이터와 초평면의 간격이 1보다 작아지게 될 때 (label = 1): e_i < 0, (label = -1): e_i > 0
                # 데이터와 초평면의 간격이 너무 멀어지게 될 때 (label = 1): e_i >> 0, (label = -1): e_i << 0
                if ((self.labels[i] * e_i < -self.tolerance) and (self.alphas[i] < self.c)) or \
                        ((self.labels[i] * e_i > self.tolerance) and (self.alphas[i] > 0)):
                    # random하게 j를 선택
                    j = select_j_rand(i, self.m)
                    fx_j = float(
                        np.multiply(self.alphas, self.labels).T * self.features * self.features[j, :].T + self.b)
                    e_j = fx_j - float(self.labels[j])
                    alpha_i_old = self.alphas[i].copy()
                    alpha_j_old = self.alphas[j].copy()
                    # alphas의 값이 0과 c사이에 있도록 한다.
                    if (self.labels[i] != self.labels[j]):
                        l = max(0, self.alphas[j] - self.alphas[i])
                        h = min(self.c, self.c + self.alphas[j] - self.alphas[i])
                    else:
                        l = max(0, self.alphas[j] + self.alphas[i] - self.c)
                        h = min(self.c, self.alphas[j] + self.alphas[i])

                    if l == h:
                        print "l==h"
                        continue

                    # 2dot(x_i, x_j) - dot(x_i, x_i) - dot(x_j, x_j)
                    eta = 2.0 * self.features[i, :] * self.features[j, :].T - \
                          self.features[i, :] * self.features[i, :].T - \
                          self.features[j, :] * self.features[j, :].T

                    if eta >= 0:
                        print "eta>0"
                        continue

                    # update alphas[j]
                    self.alphas[j] -= self.labels[j] * (e_i - e_j) / eta
                    self.alphas[j] = clipAlpha(self.alphas[j], h, l)

                    if (abs(self.alphas[j] - alpha_j_old) < 0.00001):
                        print "j not moving enough"
                        continue
                    # update alphas[j]
                    self.alphas[i] += self.labels[j] * self.labels[i] * (alpha_j_old - self.alphas[j])

                    b1 = self.b - e_i - self.labels[i] * (self.alphas[i] - alpha_i_old) * \
                         self.features[i, :] * self.features[i, :].T - \
                         self.labels[j] * (self.alphas[j] - alpha_j_old) * \
                         self.features[i, :] * self.features[j, :].T

                    b2 = self.b - e_j - self.labels[i] * (self.alphas[i] - alpha_i_old) * \
                         self.features[i, :] * self.features[j, :].T - \
                         self.labels[j] * (self.alphas[j] - alpha_j_old) * \
                         self.features[j, :] * self.features[j, :].T

                    if (0 < self.alphas[i]) and (self.c > self.alphas[i]):
                        self.b = b1
                    elif (0 < self.alphas[j]) and (self.c > self.alphas[j]):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    alpha_pairs_changed += 1
                    print "iterations : %d i: %d, pairs changed %d" %(self.iter, i, alpha_pairs_changed)

            if (alpha_pairs_changed == 0):
                self.iter += 1
            else:
                self.iter = 0

            print "iterations number: %d" %self.iter
        return self.b, self.alphas

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