import numpy as np


def rot90(imgs, gt_imgs):
    imgs_90 = np.asarray([np.rot90(imgs[i]) for i in range(len(imgs))])
    gt_imgs_90 = np.asarray([np.rot90(gt_imgs[i]) for i in range(len(gt_imgs))])
    return imgs_90, gt_imgs_90

def rot180(imgs, gt_imgs):
    imgs_180 = np.asarray([np.rot90(imgs[i], 2) for i in range(len(imgs))])
    gt_imgs_180 = np.asarray([np.rot90(gt_imgs[i], 2) for i in range(len(gt_imgs))])
    return imgs_180, gt_imgs_180

def rot270(imgs, gt_imgs):
    imgs_270 = np.asarray([np.rot90(imgs[i], 3) for i in range(len(imgs))])
    gt_imgs_270 = np.asarray([np.rot90(gt_imgs[i], 3) for i in range(len(gt_imgs))])
    return imgs_270, gt_imgs_270

def fliph(imgs, gt_imgs):
    imgs_flipped = np.asarray([np.flipud(imgs[i]) for i in range(len(imgs))])
    gt_imgs_flipped = np.asarray([np.flipud(gt_imgs[i]) for i in range(len(gt_imgs))])
    return imgs_flipped, gt_imgs_flipped

def flipv(imgs, gt_imgs):
    imgs_flipped = np.asarray([np.fliplr(np.flipud(imgs[i])) for i in range(len(imgs))])
    gt_imgs_flipped = np.asarray([np.fliplr(np.flipud(gt_imgs[i])) for i in range(len(gt_imgs))])
    return imgs_flipped, gt_imgs_flipped

def flipvh(imgs, gt_imgs):
    imgs, gt = fliph(imgs, gt_imgs)
    imgs, gt = flipv(imgs, gt)
    return imgs, gt

def augment_data(imgs, gt_imgs):
  imgs_90, gt_imgs_90 = rot90(imgs, gt_imgs)
  imgs_180, gt_imgs_180 = rot180(imgs, gt_imgs)
  imgs_270, gt_imgs_270 = rot270(imgs, gt_imgs)
  imgs_flipped, gt_imgs_flipped = fliph(imgs, gt_imgs)
  imgs_flipped2, gt_imgs_flipped2 = flipv(imgs, gt_imgs)
  imgs = np.concatenate((imgs, imgs_90, imgs_180, imgs_270, imgs_flipped, imgs_flipped2))
  gt_imgs = np.concatenate((gt_imgs, gt_imgs_90, gt_imgs_180, gt_imgs_270, gt_imgs_flipped, gt_imgs_flipped2))
  return imgs, gt_imgs
