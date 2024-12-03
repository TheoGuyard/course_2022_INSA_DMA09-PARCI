import numpy as np
from scipy.fftpack import dct
from numpy.matlib import repmat


def genDCT(dims, fact):
    """
    Generates Discrete Consine Transform dictionary.
    
    Arguments
    ---------
    dims : np.ndarray of length 3
        Overcompletness of DCT matrix (keep entries at 1 or 2).
    fact : np.ndarray of length 3
        Patch dimensions. Third dimensions has to be 3 (this function is 
        designed for RGB images or masks of missing data for RGB images).

    Credits to Jeremy Cohen.
    """

    # Initialisation of the dictionary
    # Dictionary sizes
    di = fact * dims
    # Generating the DCT matrices
    D1 = dct(np.eye(di[0]))
    D2 = dct(np.eye(di[1]))
    D3 = dct(np.eye(di[2]))
    # Truncating the DCT matrices
    D1 = D1[0 : dims[0], :]
    D2 = D2[0 : dims[1], :]
    D3 = D3[0 : dims[2], :]
    # Normalizing after truncation
    D1 = D1 * repmat(1 / np.sqrt(np.sum(D1**2, 0)), dims[0], 1)
    D2 = D2 * repmat(1 / np.sqrt(np.sum(D2**2, 0)), dims[1], 1)
    D3 = D3 * repmat(1 / np.sqrt(np.sum(D3**2, 0)), dims[2], 1)
    # Creating the big dictionary (already normalized)
    Do = np.kron(np.kron(D3, D2), D1)
    return Do


def patches(image, dims, skip):
    """
    This function computes a patch matrix Y from image, which is a (RGB) color
    image stored in a 3-way array. The user can specify the size of the patches
    dims (the third dimensions has to be 3), and the
    overlap of patches by specifying the amount of shift pixelwise between each
    patch.


    Arguments
    ---------
    image : np.ndarray
        Array containing the image. May be a mask of missing data.
    dims : np.ndarray of length 3
        Dimensions of the patches. Example: [10,10,3]. Third dimensions has to
        be 3 (this function is designed for RGB images or masks of missing data 
        for RGB images).

    Return
    ------
    Y : np.ndarray
        Matrix containing vectorized patches column-wise.

    Credits to Jeremy Cohen. 
    """

    # Sanity check
    if dims[2] != 3:
        print("dims[2] has to be 3")
        return

    # Parameters
    m1, m2, _ = image.shape

    # Initialisation of Y
    Y = np.zeros([np.prod(dims), 1])

    # Initialisation of counters
    x = 0
    y = 0

    # Building the patches
    while x + dims[0] <= m1:
        while y + dims[1] <= m2:
            impatch = image[x : (x + dims[0]), y : (y + dims[1]), :]
            Y = np.concatenate((Y, np.reshape(impatch, [np.prod(dims), 1])), axis=1)
            y = y + skip
        x = x + skip
        y = 0
    Y = np.double(Y)
    # Removing the first zero column (dirty trick)
    Y = Y[:, 1:]
    return Y


def patch2image(Y, dims, skip, imdim):
    """
    This function computes an image from a patch matrix Y, using the average of 
    the values of the same pixel when they belong to several patches.

    Arguments
    ---------
    Y : np.ndarray
        Numpy array containing the patches. Cannot have missing data.
    dims : np.ndarray of length 3
        Dimensions of the patches. Example: [10,10,3]. Third dimensions has to
        be 3 (this function is designed for RGB images or masks of missing data 
        for RGB images).
    skip : int
        Amount of shift when building the patches.
    imdim : np.ndarray of size 3
        Dimensions of the image. Example: [10,10,3]. Third dimensions has to
        be 3 (this function is designed for RGB images or masks of missing data 
        for RGB images).

    Return
    ------
    image : np.ndarray 
        Array containing RGB intensities between 1 and 255. Pixel values are
        averaged over shared patches and floored.

    Credits to Jeremy Cohen.
    """

    # Sanity check
    if dims[2] != 3:
        print("dims[2] has to be 3")
        return

    # Initialisation of the image
    image = np.zeros(imdim)
    meancount = np.zeros(imdim)

    # Initialisation of counters
    x = 0
    y = 0
    p = 0

    # Building the image with superpositions
    while x + dims[0] <= imdim[0]:
        while y + dims[1] <= imdim[1]:
            # Fetching the patch
            impatch = np.reshape(Y[:, p], dims)
            # adding the patch to the image
            image[x : (x + dims[0]), y : (y + dims[1]), :] = (
                image[x : (x + dims[0]), y : (y + dims[1]), :] + impatch
            )
            # counting how many times each pixel has been written on
            meancount[x : (x + dims[0]), y : (y + dims[1]), :] = (
                meancount[x : (x + dims[0]), y : (y + dims[1]), :] + 1
            )
            # counter
            y = y + skip
            p += 1
        x = x + skip
        y = 0

    # Averaging, clipping and rounding
    image = np.uint8(
        np.abs(image / meancount) / np.max(np.abs(image / meancount)) * 255
    )
    # image = np.uint8(np.maximum(np.minimum(np.floor(image/meancount),255), 0))
    return image
