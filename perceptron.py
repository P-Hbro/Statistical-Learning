import pandas as pd
import numpy as np
import cv2
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Preceptron(object):
    def __init__(self):
        self.learning_step = 0.0001
        self.max_iteration = 5000

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0])+1)#list * n 生成多维数组，使用的是浅拷贝
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = np.random.randint(0, len(labels)-1)#?
            x = list(features[index])#?
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum([self.w[j]*x[j] for j in range(len(self.w))])
            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue


