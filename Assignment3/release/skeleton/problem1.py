from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gco
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from utils import rgb2gray
import math


def edges4connected(height, width):
    """ Construct edges for 4-connected neighborhood MRF. Assume row-major ordering.

      Args:
        height of MRF.
        width of MRF.

      Returns:
        A `nd.array` with dtype `int32/int64` of size |E| x 2.
    """

    # construct a matrix filled with indices
    edges = []
    height_edges = []
    width_edges = []
    for i in range(height):
        for j in range(width):
            width_edges.append([[i, j], [i, j + 1]])
            height_edges.append([[i, j], [i + 1, j]])


    edges = np.hstack((width_edges, height_edges))

    return edges


def mrf_denoising_nllh(x, y, sigma_noise):
    """ Elementwise stereo negative log likelihood.

      Args:
        x                    candidate denoised image
        y                    noisy image
        sigma_noise:        noise level for Gaussian noise

      Returns:
        A `nd.array` with dtype `float32/float64`.
    """
    x = x.reshape(y.shape)
    nllh = np.sum(1/(2*sigma_noise**2) * (x-y)**2)
    assert (nllh.dtype in [np.float32, np.float64])
    return nllh


def alpha_expansion(i0, edges, i1_0, candidate_pixel_values, s, lmbda):
    """ Run alpha-expansion algorithm.

      Args:
        i0:                  Given grayscale image.
        edges:                   Given neighboor of MRF.
        i1_0:                      Initial denoised image.
        candidate_pixel_values:   Set of labels to consider
        lmbda:                   Regularization parameter for Potts model.

      Runs through the set of candidates and iteratively expands a label.
      If there have been recorded changes, re-run through the complete set of candidates.
      Stops, if there are no changes anymore.

      Returns:
        A `nd.array` of type `int32`. Assigned labels minimizing the costs.


    """
    i1 = i1_0.reshape(-1)

    for i in range(candidate_pixel_values.shape[0]):
        i1_copy = []
        candidate =[]
        unary = np.zeros([i1.shape[0], 2])
        for j in range(i1.shape[0]):
            candidate = i1
            candidate[i] = candidate_pixel_values[i]
            unary[j][0] = mrf_denoising_nllh(i1, i0, s)
            unary[j][1] = mrf_denoising_nllh(candidate, i0, s)

    i1 = i1.reshape(i0.shape)
    return i1


def show_images(i0, i1):
    """
    Visualize estimate and ground truth in one Figure.
    """
    row, col = np.nonzero(i0)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(i0, "gray", interpolation='nearest')
    plt.subplot(1,2,2)
    plt.imshow(i1, "gray", interpolation='nearest')
    plt.show()


def compute_psnr(img1, img2):
    """Computes psnr of groundtruth image and the final denoised image."""
    mse = np.mean((img1 -img2) **2)
    v_max = np.max(img1)
    psnr = 10* np.log10((v_max**2) / mse)
    return psnr

def problem1():
    # Read images
    i0 = ((255 * plt.imread('la-noisy.png')).squeeze().astype(np.int32)).astype(np.float32)
    gt = (255 * plt.imread('la.png')).astype(np.int32)
    lmbda = 5.0
    s = 5.0 

    # Create 4 connected edge neighborhood
    edges = edges4connected(i0.shape[0], i0.shape[1])
    # Candidate search range
    candidate_pixel_values = np.arange(0, 255)

    # Graph cuts with random initialization
    random_init = np.random.randint(low=0, high=255, size=i0.shape)
    estimate_random_init = alpha_expansion(i0, edges, random_init, candidate_pixel_values, s, lmbda)
    show_images(i0, estimate_random_init)
    psnr_before = compute_psnr(i0, gt)
    psnr_after = compute_psnr(estimate_random_init, gt)
    print(psnr_before, psnr_after)

if __name__ == '__main__':
    problem1()
