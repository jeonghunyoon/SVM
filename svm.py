#-*- coding: utf-8 -*-

import random

class Svm:
    def __init__(self):
        self.training_set = {}

    def load_data(self, file_name):
        features = []
        labels = []

        test_set_file = open(file_name)

        for line in test_set_file.readlines():
            tokens = line.strip().split(',')
            features.append([float(tokens[0]), float(tokens[1])])
            labels.append(float(tokens[2]))

        self.training_set['features'] = features
        self.training_set['labels'] = labels

        return self.training_set

# SMO에서 사용할 i,j pair를 찾아준다. i가 고정되었을 때의 j를 구해줄 것이다.
def select_j_rand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

# alpha j 의 값을 상한과 하한값의 사이에 둔다.
def clipAlpha(a_j, h, l):
    if a_j > h:
        a_j = h
    if a_j < l:
        a_j = l
    return a_j
