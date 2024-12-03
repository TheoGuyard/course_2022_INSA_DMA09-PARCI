# Include the src directory in Python import path
import os
os.chdir(os.getcwd())

# Imports
import numpy as np
from src.preprocessing import genDCT
from src.ista import ista

def mod(Y, lmbd, dims, fact, k=20, maxit=20, tol=1e-3, verbose=True):
    """
    Iteratively construct a dictionary D and a matrix of sparse coefficients x.
    The algorithm implemented is Method of Optimal Directions (MOD), which
    alternates between
        (1)     D = argmin_D || Y - DX ||_F^2
    which is a quadratic problem, and
        (2)     x = argmin_x || Y - Dx ||_F^2 + lmbd ||x||_1
    which is solved using ISTA. Initialization is carried out with a DCT 
    dictionary.

    Arguments
    ---------
    Y : np.ndarray
        Target patch matrix.
    lmbd : float
        Regularization parameter in the ISTA problem.
    fact : np.ndarray
        Overcompletness of the initial DCT matrix.
    dims : np.ndarray
        Patch dimensions in the initial DCT matrix.
    k : int
        Sparsity level targeted.
    maxit : int
        Maximum number of MOD iterations.
    tol : float
        Error tolerance in MOD algorithm.
    verbose : bool
        Toggle verbosity.

    Returns
    -------
    D : np.ndarray
        The learned dictionary.
    X : np.ndarray
        The last sparse estimate of Y through D
    e : np.ndarray
        The error at each iterations.

    Credits to Jeremy Cohen.
    """

    if verbose:
        print("Initialization...")

    # Initialization with DCT dictionary
    D = genDCT(dims, fact)
    _, X, ei = ista(Y, D, lmbd, k=k, tol=tol)
    m, d = D.shape
    _, n = Y.shape
    e = [ei[-1]]        # error at each iteration (last ISTA error)

    it = 0

    while True:

        it += 1

        # There might be zero rows in X. In this case, any value in D is a 
        # minimizer of problem (1). Thus, we use the last computed iterate.
        nnz_idx = [p for p in range(d) if np.any(X[p, :] != 0)]

        # Solve the problem (1) using numpy least-squares method
        D[:, nnz_idx] = np.linalg.lstsq(X.T[:, nnz_idx], Y.T, rcond=None)[0].T

        # Normalize columns of D
        norms = np.sqrt(np.sum(D**2, 0))
        norms[norms < 1e-8] = 1.    # safeguard for 0-norm columns
        D = D * np.matlib.repmat(1 / norms, m, 1)

        # Solve the problem (2) with ISTA        
        _, X, ei = ista(Y, D, lmbd, k=k, tol=tol)

        # Update error vector
        e.append(ei[-1])

        # Displays
        if verbose:
            print("Iter : {}, Err : {}".format(it, e[-1]))

        # Stopping criteria
        if np.abs(e[-1] - e[-2]) / e[-1] <= tol:
            if verbose:
                print("Tol reached")
            break
        if it >= maxit:
            if verbose:
                print("Maxit reached")
            break

    return D, X, e
