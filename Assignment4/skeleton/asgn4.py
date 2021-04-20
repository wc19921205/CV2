from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from skimage import color
from scipy.sparse import csr_matrix
import scipy.optimize
import torch
from torch import optim
from torch.nn import functional as tf
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import VOC_LABEL2COLOR
from utils import VOC_STATISTICS
from utils import numpy2torch
from utils import torch2numpy
scipy.optimize.minimize()

class VOC2007Dataset(Dataset):
    def __init__(self, root, train, num_examples):
        super().__init__()
        self.num_examples = num_examples
        self.root = root
        self.train = train
        self.train_images = []
        self.train_gts = []
        self.val_images = []
        self.val_gts = []

        if train:
            train_index = []

            with open(root + 'ImageSets/Segmentation/train.txt') as train_f:
                for i, line in enumerate(train_f):
                    if i < num_examples:
                        train_index.append(line.strip('\n'))
            for index in train_index:
                ima = numpy2torch(np.array(Image.open(root + 'JPEGImages//' + index + '.jpg')))
                gt = np.array(Image.open(root + 'SegmentationClass//' + index + '.png')).reshape(-1)

                gt[gt == 255] = 0
                gt_color = numpy2torch(np.array(VOC_LABEL2COLOR)[gt].reshape((ima.shape[1], ima.shape[2], 3)))
                self.train_images.append(ima)
                self.train_gts.append(gt_color)

        if not train:
            val_index = []
            with open(root + 'ImageSets/Segmentation/val.txt') as val_f:
                for i, line in enumerate(val_f):
                    if i < num_examples:
                        val_index.append(line.strip('\n'))

            for index in val_index:
                ima = numpy2torch(np.array(Image.open(root + 'JPEGImages//' + index + '.jpg')))
                gt = np.array(Image.open(root + 'SegmentationClass//' + index + '.png')).reshape(-1)
                gt[gt == 255] = 0
                gt_color = numpy2torch(np.array(VOC_LABEL2COLOR)[gt].reshape((ima.shape[1], ima.shape[2], 3)))

                self.val_images.append(ima)
                self.val_gts.append(gt_color)

    def __getitem__(self, index):
        example_dict = dict()
        if self.train_images:
            example_dict['im'] = self.train_images[index]
            example_dict['gt'] = self.train_gts[index]

        if self.val_images:
            example_dict['im'] = self.val_images[index]
            example_dict['gt'] = self.val_gts[index]
        assert (isinstance(example_dict, dict))
        assert ('im' in example_dict.keys())
        assert ('gt' in example_dict.keys())
        return example_dict

    def __len__(self):
        return self.num_examples


def create_loader(dataset, batch_size, shuffle, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    assert (isinstance(loader, DataLoader))
    return loader


def voc_label2color(np_image, np_label):
    assert (isinstance(np_image, np.ndarray))
    assert (isinstance(np_label, np.ndarray))

    np_image = (np_image / 255)
    np_label = (np_label / 255)

    ima_hsv = color.rgb2hsv(np_image)
    lab_hsv = color.rgb2hsv(np_label)

    lab_hsv[:, :, 2] = ima_hsv[:, :, 2]
    colored = color.hsv2rgb(lab_hsv)

    assert (np.equal(colored.shape, np_image.shape).all())
    return colored


def show_dataset_examples(loader, grid_height, grid_width, title):
    colored = []
    for i, item in enumerate(loader):
        if i < grid_height * grid_width:
            np_image = torch2numpy(torch.squeeze(item['im']))
            np_label = torch2numpy(torch.squeeze(item['gt']))
            colored.append(voc_label2color(np_image, np_label))
    plt.figure(figsize=(10, 5))
    plt.suptitle(title)
    for i in range(len(colored)):
        plt.subplot(grid_height, grid_width, i + 1)
        plt.imshow(colored[i])
    plt.show()


def standardize_input(input_tensor):
    input_np = torch2numpy(input_tensor)
    normalized = (input_np - VOC_STATISTICS['mean']) / VOC_STATISTICS['std']
    normalized = numpy2torch(normalized)
    assert (type(input_tensor) == type(normalized))
    assert (input_tensor.size() == normalized.size())
    return normalized


def run_forward_pass(normalized, model):
    model.eval()
    acts = model(normalized)['out'][0]
    prediction = acts.argmax(0)

    assert (isinstance(prediction, torch.Tensor))
    assert (isinstance(acts, torch.Tensor))
    return prediction, acts


def average_precision(prediction, gt):
    accuracy = np.round(np.sum((prediction == gt)) / gt.reshape(-1).shape[0], 3)
    return accuracy


def show_inference_examples(loader, model, grid_height, grid_width, title):
    example_image = []
    avg_prec= []
    for i, item in enumerate(loader):
        if i < grid_height * grid_width:
            np_image = torch2numpy(torch.squeeze(item['im']))
            np_label = torch2numpy(torch.squeeze(item['gt']))
            colored = numpy2torch(voc_label2color(np_image, np_label))
            norm_colored = standardize_input(colored).unsqueeze(0).float()
            prediction, acts = run_forward_pass(norm_colored, model)

            prediction_colored_label = np.array(VOC_LABEL2COLOR)[prediction.numpy().reshape(-1)]
            prediction_colored_label = prediction_colored_label.reshape(np_label.shape)
            prediction_colored = voc_label2color(np_image, prediction_colored_label)
            avg_prec.append('avg_prec: '+ str(average_precision(prediction_colored_label,np_label)))
            example_image.append(np.hstack((torch2numpy(colored),prediction_colored)))

    plt.figure(figsize=(10, 5))
    for i in range(len(avg_prec)):
        plt.subplot(grid_height, grid_width, i + 1)
        plt.title(avg_prec[i])
        plt.imshow(example_image[i])
    plt.show()


def find_unique_example(loader, unique_foreground_label):
    label = list(set([VOC_LABEL2COLOR[0]]+[VOC_LABEL2COLOR[unique_foreground_label]]))
    for item in loader:
        uniq_label = list(set(map(tuple, torch2numpy(torch.squeeze(item['gt'])).reshape(-1, 3))))
        if uniq_label == label:
            break

    example = dict()
    example['im'] = item['im']
    example['gt'] = item['gt']

    assert (isinstance(example, dict))
    return example


def show_unique_example(example_dict, model):
    np_image = torch2numpy(torch.squeeze(example_dict['im']))
    np_label = torch2numpy(torch.squeeze(example_dict['gt']))
    colored = numpy2torch(voc_label2color(np_image, np_label))
    norm_colored = standardize_input(colored).unsqueeze(0).float()
    prediction, acts = run_forward_pass(norm_colored, model)

    prediction_colored_label = np.array(VOC_LABEL2COLOR)[prediction.numpy().reshape(-1)]
    prediction_colored_label = prediction_colored_label.reshape(np_label.shape)
    prediction_colored = voc_label2color(np_image, prediction_colored_label)
    avg_prec = 'avg_prec: ' + str(average_precision(prediction_colored_label, np_label))
    example_image = np.hstack((torch2numpy(colored), prediction_colored))
    plt.imshow(example_image)
    plt.title(avg_prec)
    plt.show()
    pass


def show_attack(example_dict, model, src_label, target_label, learning_rate, iterations):
    ima = torch2numpy(torch.squeeze(example_dict['im']))
    src_gt = torch2numpy(torch.squeeze(example_dict['gt']))
    arc_label = [VOC_LABEL2COLOR[0], VOC_LABEL2COLOR[src_label]]
    # target_label = [VOC_LABEL2COLOR[0], VOC_LABEL2COLOR[target_label]]
    fake_gt = target_label * ((src_gt.reshape(-1, 3) == arc_label[1]).all(1)).astype(int)
    fake_gt = numpy2torch(fake_gt.reshape(src_gt.shape[0], src_gt.shape[1]))
    fake_gt_colored = np.array(VOC_LABEL2COLOR)[fake_gt].reshape(src_gt.shape)

    colored = numpy2torch(voc_label2color(ima, src_gt))
    norm_colored = standardize_input(colored).unsqueeze(0).float()
    prediction, acts = run_forward_pass(norm_colored, model)
    # prediction_colored_label = np.array(VOC_LABEL2COLOR)[prediction.numpy().reshape(-1)]
    # prediction_colored_label = numpy2torch(prediction_colored_label.reshape(src_gt.shape))


    cross_entropy = torch.nn.CrossEntropyLoss()

    input_tensor = torch.autograd.Variable(numpy2torch(ima).float(), requires_grad=True)
    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)


    print(input_tensor.shape)
    model.eval()
    for epoch in range(iterations):
        optimizer.zero_grad()

        input_tensor.requires_grad = False
        colored = numpy2torch(voc_label2color(torch2numpy(input_tensor), fake_gt_colored.astype(float)))
        input_tensor.requires_grad = True

        norm_colored = standardize_input(colored).unsqueeze(0).float()
        prediction, acts = run_forward_pass(norm_colored, model)

        loss = cross_entropy(acts.unsqueeze(0), fake_gt.long())
        print(input_tensor.grad)
        print('epoch: ', epoch)
        print('loss: ', loss)
        loss.backward()
        optimizer.step()



    pass


def asgn4():
    # Please set an environment variables 'VOC2007_HOME' pointing to your '../VOCdevkit/VOC2007' folder
    os.environ["VOC2007_HOME"] = './VOCdevkit/VOC2007/'
    root = os.environ["VOC2007_HOME"]
    # create datasets for training and validation
    train_dataset = VOC2007Dataset(root, train=True, num_examples=128)
    valid_dataset = VOC2007Dataset(root, train=False, num_examples=128)

    # create data loaders for training and validation
    # you can safely assume batch_size=1 in our tests..
    train_loader = create_loader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = create_loader(valid_dataset, batch_size=1, shuffle=True, num_workers=1)

    # show some images for the training and validation set
    show_dataset_examples(train_loader, grid_height=2, grid_width=3, title='training examples')
    show_dataset_examples(valid_loader, grid_height=2, grid_width=3, title='validation examples')

    # Load FCN network
    model = models.segmentation.fcn_resnet101(pretrained=True, num_classes=21)

    # Apply FCN. Switch to training loader if you want more variety.
    show_inference_examples(valid_loader, model, grid_height=2, grid_width=3, title='inference examples')

    # attack1: convert cat to dog
    cat_example = find_unique_example(valid_loader, unique_foreground_label=8)
    show_unique_example(cat_example, model=model)
    show_attack(cat_example, model, src_label=8, target_label=12, learning_rate=1.0, iterations=10)

    # feel free to try other examples..


if __name__ == '__main__':
    asgn4()
