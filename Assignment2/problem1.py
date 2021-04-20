from PIL import Image
import numpy as np


def rgb2gray(rgb):
    # TODO: convert rgb image to grayscale
    gray = rgb.convert('L')
    return gray

def load_data():
    #TODO: load images in the order i0, i1, gt
    i_0 = Image.open('i0.png')
    i_0 = rgb2gray(i_0)
    i_0 = np.asanyarray(i_0) / 255.0  # convert to numpy array and ensure pixel range between[0, 1]

    i_1 = Image.open('i1.png')
    i_1 = rgb2gray(i_1)
    i_1 = np.asanyarray(i_1) / 255.0

    g_t = Image.open('gt.png')
    g_t = np.asanyarray(g_t)
    assert np.amax(g_t) <= 16
    return i_0, i_1, g_t


def random_disparity(disparity_size):
    # create random disparity in [0,14] of size DISPARITY_SIZE
    disparity_map = np.random.randint(0, 15, size=disparity_size)
    return disparity_map


def constant_disparity(disparity_size):
    # create constant disparity of all 8's of size DISPARITY_SIZE
    disparity_map = 8 * np.ones(disparity_size)
    return disparity_map


def log_gaussian(x, mu=0, sigma=0.8):
    # Evaluate log of Gaussian (un normalized) distribution.
    # Set sigma=0.8 and mu=0.0
    value = -((x-mu)**2/(2*sigma**2))
    return value


def mrf_log_prior(x, mu=0, sigma=0.8):
    # Evaluate pairwise MRF log prior with Gaussian distributions.
    # Set sigma=1.0 and mu=0.0#
    x_j_plus_one = x[:, 1:]
    x_i_plus_one = x[1:, :]

    x_j = x[:, :-1]
    x_i = x[:-1, :]
    x_H = x_j_plus_one - x_j  # di,j+1 - di,j
    x_V = x_i_plus_one - x_i  # di+1,j - di,j
    logp = np.sum(log_gaussian(x_H, mu=0, sigma=0.8)) + np.sum(log_gaussian(x_V, mu=0, sigma=0.8))

    # x_i = x[:, :-1]
    # x_j = x[:-1, :]
    # x_i_plus_one = x[:, 1:]
    # x_j_plus_one = x[1:, :]
    #
    # value_H = log_gaussian(x_i_plus_one - x_i, mu=0, sigma=0.8)
    # value_V = log_gaussian(x_j_plus_one - x_j, mu=0, sigma=0.8)
    #
    # logp = np.sum(value_H) + np.sum(value_V)
    return logp


if __name__ == "__main__":
    i0, i1, gt = load_data()
    # Display log prior of GT disparity map

    print('log prior of GT:', mrf_log_prior(gt))

    # Display log prior of random disparity map
    disparity_size = gt.shape
    ran_dis = random_disparity(disparity_size)
    print('log prior of random disparity:', mrf_log_prior(ran_dis))

    # Display log prior of constant disparity map
    cons_dis = constant_disparity(disparity_size)
    print('log prior of constant disparity:', mrf_log_prior(cons_dis))

