import numpy as np
from sklearn.datasets import make_classification
from sklearn.mixture import GMM
from sklearn.preprocessing import StandardScaler
from sklearn import svm


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def power_normalize(xx, alpha=0.5):
    """Computes a alpha-power normalization for the matrix xx."""
    return np.sign(xx) * np.abs(xx) ** alpha


def L2_normalize(xx):
    """L2-normalizes each row of the data xx."""
    Zx = np.sum(xx * xx, 1)
    xx_norm = xx / np.sqrt(Zx[:, np.newaxis])
    xx_norm[np.isnan(xx_norm)] = 0
    return xx_norm


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.

    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors

    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.

    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.

    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf

    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covars_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def main():
    # Short demo.
    K = 3
    N = 600

    xx, _ = make_classification(n_samples=N, n_features=14)
    xx_tr, xx_te = xx[: -100], xx[-100:]

    print('Data normalization.')
    scaler = StandardScaler()
    # train normalization
    xx_tr = scaler.fit_transform(xx_tr)
    xx_tr = power_normalize(xx_tr, 0.5)
    xx_tr = L2_normalize(xx_tr)
    # test normalization
    xx_te = scaler.fit_transform(xx_te)
    xx_te = power_normalize(xx_te, 0.5)
    xx_te = L2_normalize(xx_te)

    gmm = GMM(n_components=K, covariance_type='diag')
    gmm.fit(xx_tr)

    fv_tr = fisher_vector(xx_tr, gmm)
    fv_te = fisher_vector(xx_te, gmm)

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y) 
    

    #pdb.set_trace()


if __name__ == '__main__':
    main()
