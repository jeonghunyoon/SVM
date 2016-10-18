# location : "/Users/yoonjeonghun/Desktop/machinelearninginaction/Ch06/testSet.txt"
def load_data(file_name):
    data_matrix = []
    labels = []

    test_set_file = open(file_name)

    for line in test_set_file.readlines():
        tokens = line.strip().split('\t')
        data_matrix.append([float(tokens[0]), float(tokens[1])])
        labels.append(float(tokens[2]))

    return data_matrix, labels