import math

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from LIB.plot import plot


def std_agg(cnt, s1, s2):
    print(cnt, s1, s2)
    return math.sqrt((s2 / cnt) - (s1 / cnt) ** 2)


class DecisionTree:
    def __init__(self, x, y, idxs=None, min_leaf=2):
        if idxs is None:
            idxs = np.arange(len(y))
        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
        self.n, self.c = len(idxs), x.shape[1]
        self.mean = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()

        # self.var_idx = None
        # self.split = None

    def find_varsplit(self):
        for i in range(self.c):
            self.find_better_split(i)
        if self.score == float('inf'):
            return
        x = self.split_col()
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs])
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])

    def find_better_split(self, var_idx):
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y ** 2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.

        for i in range(0, self.n - self.min_leaf - 1):
            xi, yi = sort_x[i], sort_y[i]
            lhs_cnt += 1;
            rhs_cnt -= 1
            lhs_sum += yi;
            rhs_sum -= yi
            lhs_sum2 += yi ** 2;
            rhs_sum2 -= yi ** 2
            if i < self.min_leaf or xi == sort_x[i + 1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std * lhs_cnt + rhs_std * rhs_cnt
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi

    @property
    def split_name(self):
        return self.x.columns[self.var_idx]

    # @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf')

    def __repr__(self):
        s = f'n: {self.n}; mean:{self.mean}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.mean
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)


N = 1
X, y = make_regression(n_samples=10, n_features=N, random_state=1)
plot(X.reshape(-1), y, file="a.png")
X = pd.DataFrame(X)

kol_tree = 30
xi = X
yi = y
ei = 0
n = len(yi)
pred_f = 0

for i in range(kol_tree):
    tree = DecisionTree(xi, yi)
    tree.find_better_split(0)

    print(tree.split)
    r = np.where(xi == tree.split)[0][0]
    print(r)

    left_idx = np.where(xi <= tree.split)[0]
    right_idx = np.where(xi > tree.split)[0]

    predi = np.zeros(n)
    np.put(predi, left_idx, np.repeat(np.mean(yi[left_idx]), r))  # replace left side mean y
    np.put(predi, right_idx, np.repeat(np.mean(yi[right_idx]), n - r))  # right side mean y

    predi = predi[:, None]
    pred_f = pred_f + predi

    ei = y - pred_f
    yi = ei
