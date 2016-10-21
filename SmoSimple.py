# -*- coding: utf-8 -*-

import numpy as np

class SmoSimple:
    def __init__(self, features, labels, c, tolerance, max_iterations):
        self.features = np.matrix(features).transpose()
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
        print(self.alphas)

    def run(self):
        while (self.iter < self.max_iterations):
            alpha_pairs_changed = 0
            for i in range(self.m):
                # model의 결과값
                fx_i = float(np.multiply(self.alphas, self.labels).T * self.features * self.features[i, :].T + self.b)
                # error 값
                e_i = fx_i - float(self.labels[i])
