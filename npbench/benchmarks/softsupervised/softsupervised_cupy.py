import cupy as np
import sys
from sklearn.metrics.pairwise import euclidean_distances


def local_rbfdot(dat, quantile=0.02):
    (N,d) = dat.shape

    # Normalize data columns between [-1,1]
    means = np.mean(dat, 0)
    dat = np.subtract(dat, means)
    maxs = np.amax(np.absolute(dat), 0)
    maxs = [maxs[i] if maxs[i] > np.finfo(float).eps else 1.0 for i in range(len(maxs))]
    dat = np.divide(dat, maxs)
    kern = euclidean_distances(dat, dat, squared=True)

    # compute element specific standard deviations
    sigmas = np.array([np.percentile(np.sqrt(kern), quantile*100, axis=1)])
    c_sigmas = sigmas.reshape((-1,1))

    # external dot product for direct rbfdot division
    c_sigmas = c_sigmas.dot(sigmas)
    kern = np.exp(-kern/c_sigmas)
    return kern


class SoftSupervised():
    def __init__(self, quantile=0.02, dotfunc=local_rbfdot, alpha=2.0, mu=1e-3, nu=1e-3, max_iter=1000, tol=0.001, debug=False):
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

        # identify labeled set in training data
        self.l_inds_ = np.arange(len(self.classes_))[self.classes_ != -1]

        if len(self.l_inds_) == self.n_:
            raise Exception("classes should have at least 1 missing element (value == -1)")

        # account for special modality "-1"
        self.K_ = len(np.unique(self.classes_)) - 1

        # classes should range from -1 to K-1
        if not all(sorted(np.unique(self.classes_)) == np.arange(-1, self.K_)):
            raise Exception("classes should range from -1 to K-1")

        # initialize p, q and r, unless already exist
        if self.p_ is None:
            self.p_ = np.ones((self.n_, self.K_)) * (1./self.K_)
            self.q_ = np.ones((self.n_, self.K_)) * (1./self.K_)
            self.r_ = np.zeros((self.n_, self.K_))

        # convert labeled set to one-hot encoring in r (0 remains otherwise)
        classes_bin = self.classes_[self.l_inds_]
        classes_bin = np.array(list(zip(self.l_inds_, classes_bin)), dtype=np.int32)
        self.r_[classes_bin[:,0], classes_bin[:,1]] = 1.

        # initialize weights, add alpha regularization
        self.weights_ = self.dotfunc_(self.X_, quantile=self.quantile_)
        self.weights_ = self.weights_ + np.eye(self.n_) * self.alpha_

        # Alternating Minimization (AM) main loop
        # see Section 3 in Subramanya and Bilmes 2008
        self._update_c()
        self.current_c_ = self.new_c_
        is_finished = False
        self.n_iter_ = 0
        while not is_finished:
            self._update_p()
            self._update_q()
            self._update_c()
            self.n_iter_ += 1

            if (self.n_iter_ == self.max_iter_) or (self.current_c_ - self.new_c_ < self.tol_):
                is_finished = True
            else:
                self.current_c_ = self.new_c_


    # KL divergence between weight vectors summing to 1
    def _kl(self, p1, p2):
        p1[p1==0.] = sys.float_info.epsilon # add epsilon to avoid log(0)
        p2[p2==0.] = sys.float_info.epsilon # add epsilon to avoid division by 0
        return np.sum(p1 * np.log(p1 / p2), axis=1)

    # entropy of a weight vector
    def _h(self, p1):
        p1[p1==0] = sys.float_info.epsilon # add epsilon to avoid log(0)
        return -np.sum(p1 * np.log(p1), axis=1)

    # update objective function to be minimized
    # with current p and q
    def _update_c(self):
        res = 0.
        res += np.sum(self._kl(self.r_[self.l_inds_,:], self.q_[self.l_inds_,:]))
        for i in range(self.n_):
            res += self.mu_ * np.sum(self.weights_[i,:] * self._kl(self.p_[i,:], self.q_))
        res -= self.nu_ * np.sum(self._h(self.p_))
        self.new_c_ = res

    # update equations according to end of Section 3
    def _update_p(self):
        # additional casts for csr_matrix compatibility
        gamma = self.nu_ + self.mu_ * np.asarray(np.sum(self.weights_, axis=1)).reshape(-1) # axis=1 sums wrt lines
        gamma = gamma.reshape((-1,1))

        # add small constant to avoid log(0)
        q_epsilon = np.copy(self.q_)
        q_epsilon[q_epsilon == 0.] = sys.float_info.epsilon
        q_log = np.log(q_epsilon) - 1.

        beta = - np.ones((self.n_, self.K_)) * self.nu_
        for i in range(self.n_):
            beta[i,:] += np.sum(self.mu_ * self.weights_[i,:].reshape(-1,1) * q_log, axis=0)

        # compute and normalize p
        self.p_ = np.exp(beta / gamma)
        self.p_ = self.p_ / np.sum(self.p_, axis=1)[:,None]

    # update equations according to end of Section 3
    def _update_q(self):
        numerator = np.copy(self.r_)
        for i in range(self.n_):
            numerator[i,:] += np.sum(self.mu_ * self.weights_[:,i].reshape(-1,1) * self.p_, axis=0)

        denominator = np.sum(self.r_, axis=1) + self.mu_ * np.asarray(np.sum(self.weights_, axis=0)).reshape(-1)

        self.q_ = numerator / denominator[:,None]

    # get max_index per row of p_
    def predict(self):
        return np.argmax(self.p_, axis=1)

    # return raw p_
    def predict_proba(self):
        return self.p_

def main(data, labels):
    model = SoftSupervised(dotfunc=local_rbfdot, debug=True)
    model.fit(data, labels)
