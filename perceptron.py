import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 感知机模型（对偶形式*）
class Perceptron(object):
    def __init__(self):
        self.learning_step = 0.0001
        self.max_iteration = 5000

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)  # list * n 生成多维数组，使用的是浅拷贝
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = np.random.randint(0, len(labels) - 1)  # ?
            x = list(features[index])  # ?
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.x[i] * wx * self.learning_step

    def __predict(self, x):
        y = sum([self.w[i] * x[i] for i in range(len(self.w))])
        return int(y > 0)

    def predict(self, features):
        labels = []
        for i in range(len(features)):
            x = list(features[i])
            x.append(1.0)
            labels.append(self.__predict(x))
        return labels


if __name__ == '__main__':
    print('Start read data')

    time_1 = time.time()
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print('Start predicting')
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy socre is ", score)
