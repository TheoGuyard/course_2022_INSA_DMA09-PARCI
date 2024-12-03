import numpy as np
from scipy.linalg import svdvals

def soft_threshold(x, gamma):
    return np.maximum(np.abs(x) - gamma, 0) * np.sign(x)


def ista(Y, D, lmbd, M=None, k=20, maxit=50, tol=1e-3, verbose=False):
    """
    Solve the BPDN problem using the ISTA algorithm.

    Arguments
    ---------
    Y : np.ndarray
        Target patch matrix.
    D : np.ndarray
        Dictionary matrix. Must be normalized column-wise.
    lmbd : float
        Regularization parameter.
    M : np.ndarray
        Boolean array indicating the location of missing pixels. Zero are 
        missing pixels. Set None if there are no missing pixels.
    k : int
        Sparsity level targeted in x. If k >= 1, only the k largest 
        coefficients in x are kept and are projected onto the subspace in a 
        post-processing step. Set k = 0 for no final projection.
    maxit : int
        Maximum number of iterations.
    tol : float
        Error tolerance.
    verbose : bool
        Toggle verbosity.

    Returns
    -------
    X_ista : np.ndarray
        Sparse vector estimated from y.
    X : np.ndarray
        X_ista with only the k largest components kept and projected onto the 
        subspace in a post-processing step (X = X_ista when k = 0).
    e : np.ndarray
        Relative reconstruction error at each iteration.

    Credits to Jeremy Cohen.
    """

    # --- Checks --- #

    assert len(Y.shape) == 2
    assert len(D.shape) == 2
    assert D.shape[0] == Y.shape[0]
    assert np.all(np.array([lmbd, k, maxit, tol]) >= 0.)
    assert (M is None) or (M.shape == Y.shape)


    # --- Initialization --- #

    # Working values
    p, n = Y.shape
    _, d = D.shape
    L = svdvals(D.T @ D)[0]
    X = np.zeros([d, n])        # iterate
    R = np.zeros(Y.shape)       # residual R = Y - D @ X
    e = []                      # error at each iteration

    # Compute the residual column-wisely and avoid missing pixels
    if M is None:
        R = Y - D @ X
    else:                      
        for l in range(n):
            m = M[:, l]                             # l-th column mask
            R[m, l] = Y[m, l] - D[m, :] @ X[:, l]   # l-th column residual

    # Current error
    e.append(np.linalg.norm(R) / p)


    # --- Main loop --- #

    if verbose:
        print("ISTA is running...")

    it = 0
    while True:

        it += 1

        # ISTA iteration
        G = -D.T @ R
        W = X - G / L
        X = soft_threshold(W, lmbd / L)

        # Residual computation
        if M is None:
            R = Y - D @ X
        else:                      
            for l in range(n):
                m = M[:, l]
                R[m, l] = Y[m, l] - D[m, :] @ X[:, l]

        # Error computation
        e.append(np.linalg.norm(R) / p)

        # Displays
        if verbose:
            print("Iter : {}, Error : {}".format(it, e[-1]))
        
        # Stopping criteria
        if np.abs(e[-1] - e[-2]) / e[-1] <= tol:
            break
        if it >= maxit:
            break

    # --- Post-processing --- #

    # X_ista in the solution of ISTA
    # X is similar to X_ista but only k coefficients per patch are kept.
    X_ista = np.copy(X)

    if k >= 1:

        # Support estimation
        S = np.argsort(np.abs(X), 0)
        S = S[-k:, :]

        # Projection onto support for every columns in Y
        X = np.zeros(X.shape)
        for l in range(n):
            m = np.ones(p, dtype=bool) if M is None else M[:, l]
            s = S[:, l]
            x = np.linalg.lstsq(D[m, :][:, s], Y[m, l], rcond=None)[0]
            X[s, l] = x

        # Last residual computation
        if M is None:
            R = Y - D @ X
        else:                      
            for l in range(n):
                m = M[:, l]
                R[m, l] = Y[m, l] - D[m, :] @ X[:, l]

        # Last error computation
        e.append(np.linalg.norm(R) / p)

    return X_ista, X, e
