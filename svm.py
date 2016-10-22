#-*- coding: utf-8 -*-

class Svm:
    def __init__(self):
        self.training_set = {}

    def load_data(self, file_name):
        features = []
        labels = []

        test_set_file = open(file_name)

        for line in test_set_file.readlines():
            tokens = line.strip().split('\t')
            features.append([float(tokens[0]), float(tokens[1])])
            labels.append(float(tokens[2]))

        self.training_set['features'] = features
        self.training_set['labels'] = labels

        return self.training_set


