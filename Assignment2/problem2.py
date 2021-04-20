import numpy as np
import problem1
import scipy.optimize
from scipy import interpolate
from matplotlib import pyplot as plt


def log_gaussian(x, mu, sigma):
    value = -((x - mu) ** 2 / (2 * sigma ** 2))
    grad = -((x - mu) / (sigma ** 2))
    return value, grad


# Evaluate gradient of pairwise MRF log prior with Gaussian distributions.
def stereo_log_prior(x):
    # Evaluate stereo log prior.
    # Set: mu=0.0, sigma=0.8

    x = x.reshape((288, 384))
    x_pad = np.pad(x, ((1, 1), (1, 1)), 'constant', constant_values=0) # pad disparity map
    x_i_plus_one = x_pad[2:, 1:-1]  # keep height and weight is same with input
    x_j_plus_one = x_pad[1:-1, 2:]
    x_i_minus_one = x_pad[:-2, 1:-1]
    x_j_minus_one = x_pad[1:-1, :-2]

    value_H, grad_H = log_gaussian(x_j_plus_one - x, mu=0, sigma=0.8)  # get horizontal value and gradient
    value_V, grad_V = log_gaussian(x_i_plus_one - x, mu=0, sigma=0.8)
    value = np.sum(value_H) + np.sum(value_V)
    value_H_minus, grad_H_minus = log_gaussian(x-x_j_minus_one, mu=0, sigma=0.8)
    value_V_minus, grad_V_minus = log_gaussian(x-x_i_minus_one, mu=0, sigma=0.8)
    grad = grad_H + grad_V + grad_H_minus + grad_V_minus  # compute gradient


    return value, grad


def random_disparity(disparity_size):
    # create random disparity in [0,14] of size DISPARITY_SIZE
    disparity_map = np.random.randint(0, 15, size=disparity_size)
    return disparity_map


def constant_disparity(disparity_size):
    # create constant disparity of all 8's of size DISPARITY_SIZE
    disparity_map = 8 * np.ones(disparity_size)
    return disparity_map


def stereo_log_likelihood(x, im0, im1):
    # Evaluate stereo log likelihood.
    # Set: mu = 0.0, sigma = 1.2
        h, w = im1.shape  # get shape of image1
        x_axis = np.arange(0, w) # create x axis
        y_axis = np.arange(0, h)
        x = x.reshape((288, 384))
        f = interpolate.interp2d(x_axis, y_axis, im1, kind = 'cubic')  # interpolation, make im1 to a continuous image

        [x_new, y_new] = np.meshgrid(x_axis, y_axis)
        x_new = x_new-x  # for computing Horizontal image derivative at k - dk,l

        im1_value = np.zeros_like(x)
        im1_grad = np.zeros_like(x)
        for i in range(0, h):
            value, grad = log_gaussian(im0[i]-f(x_new[i], i), mu=0, sigma=0.8)
            im1_value[i, :] = value
            im1_grad[i, :] = -grad * interpolate.interp2d.__call__(f, x_new[i], i, dx=0, dy=1)

        im1_value = np.sum(im1_value)

        return im1_value, im1_grad


def stereo_log_posterior(x, im0, im1):
    # Evaluate stereo posterior
    prior, prior_grad = stereo_log_prior(x)
    likelihood, likelihood_grad = stereo_log_likelihood(x, im0, im1)
    log_posterior = prior + likelihood
    log_posterior_grad = prior_grad + likelihood_grad
    return log_posterior, log_posterior_grad


def target(x, im0, im1):
    return stereo_log_posterior(x, im0, im1)[0]


def post_grad(x, im0, im1):
    return stereo_log_posterior(x, im0, im1)[1]


def stereo(x0, im0, im1):
    # Run stereo algorithm using gradient ascent or sth similar
    if __name__ == '__main__':
        x = scipy.optimize.minimize(fun=target, x0=x0, args=(im0, im1), method='tnc', jac=post_grad)
    return x


if __name__ == "__main__":
    # use load_data() from problem1
    im0, im1, gt = problem1.load_data()
    # Display stereo: Initialized with constant 8's
    disparity_size = gt.shape
    con_dis = constant_disparity(disparity_size)
    x = stereo(con_dis, im0, im1)
    x = x.x.reshape((288, 384))
    plt.subplot(1, 3, 1)
    plt.imshow(gt)
    plt.title('gt')
    plt.subplot(1, 3, 2)
    plt.imshow(x)
    plt.title('stereo of constant')
    plt.subplot(1, 3, 3)
    plt.imshow(gt-x)
    plt.title('difference')
    plt.show()

    # Display stereo: Initialized with noise in [0,14]
    ran_dis = random_disparity(disparity_size)

    x = stereo(ran_dis, im0, im1)
    # print(ran_dis)
    x = x.x.reshape((288, 384))
    plt.subplot(1, 3, 1)
    plt.imshow(gt)
    plt.title('gt')
    plt.subplot(1, 3, 2)
    plt.imshow(x)
    plt.title('stereo of random disparity')
    plt.subplot(1, 3, 3)
    plt.imshow(gt - x)
    plt.title('difference')
    plt.show()

    # Display stereo: Initialized with gt
    x = stereo(gt, im0, im1)
    x = x.x.reshape((288, 384))
    plt.subplot(1, 3, 1)
    plt.imshow(gt)
    plt.title('gt')
    plt.subplot(1, 3, 2)
    plt.imshow(x)
    plt.title('stereo of gt')
    plt.subplot(1, 3, 3)
    plt.imshow(gt - x)
    plt.title('difference')
    plt.show()

    # Coarse to fine estimation..
