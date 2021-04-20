from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import torch
import torch.nn.functional as tf
import torch.optim as optim
import scipy.signal

from utils import flow2rgb
from utils import rgb2gray
from utils import read_flo
from utils import read_image


def numpy2torch(array):
    """ Converts 3D numpy HWC ndarray to 3D PyTorch CHW tensor."""
    assert (array.ndim == 3)
    h, w, c = array.shape
    array = array.reshape((c, h, w))
    return torch.from_numpy(array)


def torch2numpy(tensor):
    """ Convert 3D PyTorch CHW tensor to 3D numpy HWC ndarray."""
    assert (tensor.dim() == 3)
    c, h, w = tensor.shape
    tensor = tensor.reshape((h, w, c))
    tensortonumpy = tensor.numpy()
    return tensortonumpy


def load_data(im1_filename, im2_filename, flo_filename):
    """ Loads images and flow ground truth. Returns 4D tensors."""
    batch_size = 1
    im1 = read_image(im1_filename)
    im1 = rgb2gray(im1)
    im2 = read_image(im2_filename)
    im2 = rgb2gray(im2)
    flow_gt = read_flo(flo_filename)

    tensor1 = numpy2torch(im1)
    tensor1 = tensor1.expand((batch_size, tensor1.shape[0], tensor1.shape[1], tensor1.shape[2]))
    tensor2 = numpy2torch(im2)
    tensor2 = tensor2.expand((batch_size, tensor2.shape[0], tensor2.shape[1], tensor2.shape[2]))
    flow_gt = numpy2torch(flow_gt)
    flow_gt = flow_gt.expand((batch_size, flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2]))
    return tensor1, tensor2, flow_gt


def evaluate_flow(flow, flow_gt):
    """
    Evaluate the average endpoint error w.r.t the ground truth flow_gt.
    Excludes pixels, where u or v components of flow_gt have values > 1e9.
    """
    assert (flow.dim() == 4 and flow_gt.dim() == 4)
    assert (flow.size(1) == 2 and flow_gt.size(1) == 2)

    flow_gt = torch.where(flow_gt > 1e09, torch.zeros_like(flow_gt), flow_gt)  # clean invalid value
    flow = torch.where(flow > 1e09, torch.zeros_like(flow), flow)

    aepe = torch.sqrt(torch.sum((flow[:, 0, :, :] - flow_gt[:, 0, :, :]) ** 2 + (flow[:, 1, :, :] - flow_gt[:, 1, :, :]) ** 2))

    return aepe


def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth."""
    assert (im1.dim() == 4 and im2.dim() == 4 and flow_gt.dim() == 4)
    assert (im1.size(1) in [1, 3] and im2.size(1) in [1, 3] and flow_gt.size(1) == 2)

    flow_gt = torch.where(flow_gt > 1e09, torch.zeros_like(flow_gt), flow_gt)  # clean invalid value
    plt.figure(1, figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title('image 1')
    plt.imshow(im1.squeeze())
    im2_w = warp_image(im2, flow_gt)
    plt.subplot(2, 2, 2)
    plt.title('image 2 warp')
    plt.imshow(im2_w.squeeze())
    plt.subplot(2, 2, 3)
    plt.title('difference')
    plt.imshow(im1.squeeze() - im2_w.squeeze())
    plt.show()

    return


def warp_image(im, flow):
    """ Warps given image according to the given optical flow."""
    assert (im.dim() == 4 and flow.dim() == 4)
    assert (im.size(1) in [1, 3] and flow.size(1) == 2)


    def get_grid(im, flow):
        b, c, h, w = im.shape
        x_u = flow[0][0]
        y_v = flow[0][1]

        x_new = torch.arange(0, w).view(1, 1, w).expand(b, h, w).float() + x_u
        y_new = torch.arange(0, h).view(1, h, 1).expand(b, h, w).float() + y_v

        x_new = (x_new - w / 2) / (w / 2)
        y_new = (y_new - h / 2) / (h / 2)

        grids = torch.stack((x_new, y_new), dim=3)
        return grids

    grids = get_grid(im, flow)

    x_warp = tf.grid_sample(im, grids, mode='bilinear', padding_mode='zeros', align_corners=True)
    return x_warp


def energy_hs(im1, im2, flow, lambda_hs):
    """ Evalutes Horn-Schunck energy function."""
    assert (im1.dim() == 4 and im2.dim() == 4 and flow.dim() == 4)
    assert (im1.size(1) == 1 and im2.size(1) == 1 and flow.size(1) == 2)

    # sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # sobel_x = torch.from_numpy(sobel_x)
    # sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # sobel_y= torch.from_numpy(sobel_y)
    flow = torch.where(flow > 1e09, torch.zeros_like(flow), flow)  # clean invalid value

    def gradient_x(img):
        # gx = scipy.signal.convolve2d(img.squeeze(), sobel_x, mode='same')
        shift_x = torch.zeros_like(img)
        shift_x[:, :-1] = img[:, 1:]
        gx = img - shift_x
        #print(gx)
        return gx

    def gradient_y(img):
        shift_y = torch.zeros_like(img)
        shift_y[:-1, :] = img[1:, :]
        gy = img - shift_y
        #print(gy)
        return gy

    u_dx = gradient_x(flow[0][0])
    u_dy = gradient_y(flow[0][0])
    v_dx = gradient_x(flow[0][1])
    v_dy = gradient_y(flow[0][1])

    im2_w = warp_image(im2, flow)
    energy_first_term = (im2_w - im1) ** 2
    energy_second_term = lambda_hs * (u_dx ** 2 + u_dy ** 2 + v_dx ** 2 + v_dy ** 2)

    energy = torch.sum(energy_first_term + energy_second_term)

    return energy


def estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter):
    """
    Estimate flow using HS with Gradient Descent.
    Displays average endpoint error.
    Visualizes flow field.

    Returns estimated flow]
    """
    assert (im1.dim() == 4 and im2.dim() == 4 and flow_gt.dim() == 4)
    assert (im1.size(1) == 1 and im2.size(1) == 1 and flow_gt.size(1) == 2)
    flow = torch.rand_like(flow_gt, requires_grad=True)
    optimizer = torch.optim.SGD([flow], lr=learning_rate)

    for epoch in range(num_iter):
        optimizer.zero_grad()  # set gradient to zero
        loss = energy_hs(im1, im2, flow, lambda_hs)
        print('epoch: ', epoch)
        print('loss: ', loss)
        print('evaluate_flow: ', evaluate_flow(flow, flow_gt=flow_gt))
        loss.backward()
        optimizer.step()
    flow.requires_grad=False
    return flow


def problem2():
    # Loading data
    im1, im2, flow_gt = load_data("frame10.png", "frame11.png", "flow10.flo")

    # Parameters
    lambda_hs = 0.0015
    num_iter = 400
    # Warping_practice
    visualize_warping_practice(im1, im2, flow_gt)
    print('Difference between im1 and warped im2 is very small')
    energy_hs(im1, im2, flow_gt, 1)
    # Gradient descent
    learning_rate = 20
    es_flow = estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter)
    es_flow = flow2rgb(torch2numpy(es_flow.squeeze()))

    plt.imshow(es_flow)
    plt.show()
if __name__ == "__main__":
    problem2()
