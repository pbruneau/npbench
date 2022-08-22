import numpy as np
import cupy as cp
import sys
from sklearn.metrics.pairwise import euclidean_distances


def sqdist(X):
    sx = cp.sum(X ** 2, axis=1, keepdims=True)
    mat = -2 * X.dot(X.T) + sx + sx.T
    return cp.maximum(mat, 0.0)


def local_rbfdot(dat, quantile=0.02):
    N, d = dat.shape
    means = cp.mean(dat, 0)
    dat = cp.subtract(dat, means)
    maxs = cp.amax(cp.absolute(dat), 0)
    maxs = (maxs < np.finfo(float).eps) * 1.0 + (maxs >= np.finfo(float).eps
        ) * maxs
    dat = cp.divide(dat, maxs)
    kern = sqdist(dat)
    sigmas = cp.array([cp.percentile(cp.sqrt(kern), quantile * 100, axis=1)])
    c_sigmas = sigmas.reshape((-1, 1))
    c_sigmas = c_sigmas.dot(sigmas)
    kern = cp.exp(-kern / c_sigmas)
    return kern


class SoftSupervised:

    def __init__(self, quantile=0.02, dotfunc=local_rbfdot, alpha=2.0, mu=
        0.001, nu=0.001, max_iter=1000, tol=0.001, debug=False):
        self.quantile_ = quantile
        self.dotfunc_ = dotfunc
        self.alpha_ = alpha
        self.mu_ = mu
        self.nu_ = nu
        self.max_iter_ = max_iter
        self.tol_ = tol
        self.debug_ = debug
        self.X_ = None
        self.classes_ = None
        self.l_inds_ = None
        self.p_ = None
        self.q_ = None
        self.r_ = None
        self.n_iter_ = None
        self.current_c_ = None
        self.new_c_ = None
        self.weights_ = None

    def fit(self, X, classes):
        self.X_ = X
        self.n_ = self.X_.shape[0]
        self.classes_ = classes
        self.l_inds_ = cp.arange(len(self.classes_))[self.classes_ != -1]
        if len(self.l_inds_) == self.n_:
            raise Exception(
                'classes should have at least 1 missing element (value == -1)')
        self.K_ = len(cp.unique(self.classes_)) - 1
        if not all(cp.unique(self.classes_) == cp.arange(-1, self.K_)):
            raise Exception('classes should range from -1 to K-1')
        if self.p_ is None:
            self.p_ = cp.ones((self.n_, self.K_)) * (1.0 / self.K_)
            self.q_ = cp.ones((self.n_, self.K_)) * (1.0 / self.K_)
            self.r_ = cp.zeros((self.n_, self.K_))
        classes_bin = self.classes_[self.l_inds_]
        classes_bin = cp.array(list(zip(self.l_inds_, classes_bin)), dtype=
            np.int32)
        self.r_[classes_bin[:, (0)], classes_bin[:, (1)]] = 1.0
        self.weights_ = self.dotfunc_(self.X_, quantile=self.quantile_)
        self.weights_ = self.weights_ + cp.eye(self.n_) * self.alpha_
        self._update_c()
        self.current_c_ = self.new_c_
        is_finished = False
        self.n_iter_ = 0
        while not is_finished:
            self._update_p()
            self._update_q()
            self._update_c()
            self.n_iter_ += 1
            if (self.n_iter_ == self.max_iter_ or self.current_c_ - self.
                new_c_ < self.tol_):
                is_finished = True
            else:
                self.current_c_ = self.new_c_

    def _kl(self, p1, p2):
        p1[p1 == 0.0] = sys.float_info.epsilon
        p2[p2 == 0.0] = sys.float_info.epsilon
        return cp.sum(p1 * cp.log(p1 / p2), axis=1)

    def _h(self, p1):
        p1[p1 == 0] = sys.float_info.epsilon
        return -cp.sum(p1 * cp.log(p1), axis=1)

    def _update_c(self):
        res = 0.0
        res += cp.sum(self._kl(self.r_[(self.l_inds_), :], self.q_[(self.
            l_inds_), :]))
        for i in range(self.n_):
            res += self.mu_ * cp.sum(self.weights_[(i), :] * self._kl(self.
                p_[(i), :], self.q_))
        res -= self.nu_ * cp.sum(self._h(self.p_))
        self.new_c_ = res

    def _update_p(self):
        gamma = self.nu_ + self.mu_ * cp.asarray(cp.sum(self.weights_, axis=1)
            ).reshape(-1)
        gamma = gamma.reshape((-1, 1))
        q_epsilon = cp.copy(self.q_)
        q_epsilon[q_epsilon == 0.0] = sys.float_info.epsilon
        q_log = cp.log(q_epsilon) - 1.0
        beta = -cp.ones((self.n_, self.K_)) * self.nu_
        for i in range(self.n_):
            beta[(i), :] += cp.sum(self.mu_ * self.weights_[(i), :].reshape
                (-1, 1) * q_log, axis=0)
        self.p_ = cp.exp(beta / gamma)
        self.p_ = self.p_ / cp.sum(self.p_, axis=1)[:, (None)]

    def _update_q(self):
        numerator = cp.copy(self.r_)
        for i in range(self.n_):
            numerator[(i), :] += cp.sum(self.mu_ * self.weights_[:, (i)].
                reshape(-1, 1) * self.p_, axis=0)
        denominator = cp.sum(self.r_, axis=1) + self.mu_ * cp.asarray(cp.
            sum(self.weights_, axis=0)).reshape(-1)
        self.q_ = numerator / denominator[:, (None)]

    def predict(self):
        return cp.argmax(self.p_, axis=1)

    def predict_proba(self):
        return self.p_


def main(data, labels):
    model = SoftSupervised(dotfunc=local_rbfdot, debug=True)
    model.fit(data, labels)
