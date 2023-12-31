import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from road_segmentation.utils.augmentation import *

def visualize_augmented_data(src, gt):
  rot90_images, rot90_gt = rot90(src, gt)
  rot270_images, rot270_gt = rot270(src, gt)
  fliph_images, fliph_gt = fliph(src, gt)
  flipv_images, flipv_gt = flipv(src, gt)
  flipvh_images, flipvh_gt = flipvh(src, gt)
  imgs_contrast, gt_imgs_contrast = augment_contrast(src, gt)
  imgs_contrast2, gt_imgs_contrast2 = augment_contrast(src, gt, 2)
  imgs_holes, gt_imgs_holes = holes(src, gt, 5)
  fig = plt.figure(figsize=(50, 15))
  fig.add_subplot(2, 7, 1)
  plt.imshow(src[0])
  plt.title("Original Image")
  fig.add_subplot(2, 7, 2)
  plt.imshow(rot90_images[0])
  plt.title("Rotated 90")
  fig.add_subplot(2, 7, 3)
  plt.imshow(imgs_contrast[0])
  plt.title("Image contrast")
  fig.add_subplot(2, 7, 4)
  plt.imshow(rot270_images[0])
  plt.title("Rotated 270")
  fig.add_subplot(2, 7, 5)
  plt.imshow(fliph_images[0])
  plt.title("Flip Horizontal")
  fig.add_subplot(2, 7, 6)
  plt.imshow(imgs_contrast2[0])
  plt.title("Image contrast 2")
  fig.add_subplot(2, 7, 7)
  plt.imshow(imgs_holes[0])
  plt.title("Holes")
  fig.add_subplot(2, 7, 8)
  plt.imshow(gt[0])
  plt.title("Original GT")
  fig.add_subplot(2, 7, 9)
  plt.imshow(rot90_gt[0])
  plt.title("Rotated 90")
  fig.add_subplot(2, 7, 10)
  plt.imshow(gt_imgs_contrast[0])
  plt.title("Image contrast")
  fig.add_subplot(2, 7, 11)
  plt.imshow(rot270_gt[0])
  plt.title("Rotated 270")
  fig.add_subplot(2, 7, 12)
  plt.imshow(fliph_gt[0])
  plt.title("Flip Horizontal")
  fig.add_subplot(2, 7, 13)
  plt.imshow(gt_imgs_contrast2[0])
  plt.title("Image contrast 2")
  fig.add_subplot(2, 7, 14)
  plt.imshow(gt_imgs_holes[0])
  plt.title("Holes")
  plt.show()
  
def visualize_result(device, model, dataset):
  model.eval()
  pytorchDl = DataLoader(dataset, batch_size=1, shuffle=True)
  for i, (data, target) in enumerate(pytorchDl):
    data = data.squeeze()
    target = target.squeeze()
    inputs = data.to(device).unsqueeze(0)
    print(inputs.size())
    output = model.forward(inputs)
    data, target = data.cpu().numpy().transpose((1, 2, 0)), target.cpu().numpy()

    fig = plt.figure(figsize=(15, 15))
    fig.add_subplot(1, 4, 1)
    im = plt.imshow(data)
    plt.title("Original Image")
    fig.add_subplot(1, 4, 2)
    im = plt.imshow(target)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Ground Truth")
    fig.add_subplot(1, 4, 3)
    im = plt.imshow(nn.Sigmoid()(output.cpu().detach()).numpy())
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Prediction")
    plt.show()
    break