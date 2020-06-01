#подключаем необходимые библиотеки
import numpy as np
from sklearn.metrics import mean_squared_error as mse


class RegressionTree():
    def __init__(self, max_depth=3, n_epoch=10, min_size=10):
        self.max_depth = max_depth  # максимальная глубина
        self.min_size = min_size  # минимальный размер поддерева
        self.value = 0  # значение в поддереве (среднее по всем листьям)
        self.feature_idx = -1  # номер лучшего признака
        self.feature_threshold = 0  # значение лучшего признака
        self.left = None  # левый потомок
        self.right = None  # правый потомок

    def fit(self, X, y):
        self.value = y.mean()
        base_error = ((y - self.value) ** 2).sum()
        error = base_error
        flag = 0
        if self.max_depth <= 1:
            return
        dim_shape = X.shape[1]
        left_value = 0
        right_value = 0
        for feat in range(dim_shape):
            prev_error1, prev_error2 = base_error, 0
            idxs = np.argsort(X[:, feat])
            mean1, mean2 = y.mean(), 0
            sm1, sm2 = y.sum(), 0
            N = X.shape[0]
            N1, N2 = N, 0
            thres = 1
            flag2 = 0
            while thres < N - 1:
                N1 -= 1
                N2 += 1
                idx = idxs[thres]
                x = X[idx, feat]
                delta1 = (sm1 - y[idx]) * 1.0 / N1 - mean1
                delta2 = (sm2 + y[idx]) * 1.0 / N2 - mean2
                sm1 -= y[idx]
                sm2 += y[idx]
                prev_error1 += (delta1 ** 2) * N1
                prev_error1 -= (y[idx] - mean1) ** 2
                prev_error1 -= 2 * delta1 * (sm1 - mean1 * N1)
                mean1 = sm1 / N1
                prev_error2 += (delta2 ** 2) * N2
                prev_error2 += (y[idx] - mean2) ** 2
                prev_error2 -= 2 * delta2 * (sm2 - mean2 * N2)
                mean2 = sm2 / N2
                if thres < N - 1 and np.abs(x - X[idxs[thres + 1], feat]) < 1e-5:
                    thres += 1
                    continue
                if prev_error1 + prev_error2 < error:
                    if (min(N1, N2) > self.min_size):
                        self.feature_idx, self.feature_threshold = feat, x
                        left_value, right_value = mean1, mean2
                        flag = 1
                        error = prev_error1 + prev_error2
                thres += 1
        if self.feature_idx == -1:
            return
        self.left = RegressionTree(self.max_depth - 1)
        self.left.value = left_value
        self.right = RegressionTree(self.max_depth - 1)
        self.right.value = right_value
        idxs_l = (X[:, self.feature_idx] > self.feature_threshold)
        idxs_r = (X[:, self.feature_idx] <= self.feature_threshold)

        self.left.fit(X[idxs_l, :], y[idxs_l])
        self.right.fit(X[idxs_r, :], y[idxs_r])

    def __predict(self, x):


        if self.feature_idx == -1:
            return self.value

        if x[self.feature_idx] > self.feature_threshold:
            return self.left.__predict(x)
        else:
            return self.right.__predict(x)


    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = self.__predict(X[i])
        return y


class GradientBoosting():
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 random_state=17, n_samples=15, min_size=5, base_tree='Bagging'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth # глубина
        self.learning_rate = learning_rate # ставка
        self.initialization = lambda y: np.mean(y) * np.ones([y.shape[0]])
        self.min_size = min_size
        self.loss_by_iter = []
        self.trees_ = []
        self.loss_by_iter_test = []
        self.n_samples = n_samples
        self.base_tree = base_tree

    def fit(self, X, y):
        self.X = X
        self.y = y
        b = self.initialization(y)
        prediction = b.copy()
        for t in range(self.n_estimators):
            if t == 0:
                resid = y
            else:
                resid = (y - prediction)
            tree = RegressionTree(max_depth=self.max_depth, min_size=self.min_size)
            tree.fit(X, resid)
            b = tree.predict(X).reshape([X.shape[0]])
            self.trees_.append(tree)
            prediction += self.learning_rate * b
            if t > 0:
                self.loss_by_iter.append(mse(y, prediction))
        return self

    def predict(self, X):
        pred = np.ones([X.shape[0]]) * np.mean(self.y)
        for t in range(self.n_estimators):
            pred += self.learning_rate * self.trees_[t].predict(X).reshape([X.shape[0]])
        return [1 if i > 0.5 else 0 for i in pred]
