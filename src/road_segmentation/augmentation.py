import numpy as np
from scipy import ndimage
from PIL import Image, ImageEnhance
from skimage import io 
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator



def augment_contrast(imgs, gt_imgs, contrast_factor=3):
    imgs_contrast = np.asarray([np.clip(img * contrast_factor, 0, 1) for img in imgs])
    return imgs_contrast, gt_imgs

def rot45(imgs, gt_imgs):
    imgs_rot45 = np.asarray([np.clip(ndimage.rotate(img, 45, reshape=False), 0, 1) for img in imgs])
    gt_imgs_rot45 = np.asarray([np.clip(ndimage.rotate(gt_img, 45, reshape=False), 0, 1) for gt_img in gt_imgs])
    return imgs_rot45, gt_imgs_rot45

def rot135(imgs, gt_imgs):
    imgs_rot135 = np.asarray([np.clip(ndimage.rotate(img, 135, reshape=False), 0, 1) for img in imgs])
    gt_imgs_rot135 = np.asarray([np.clip(ndimage.rotate(gt_img, 135, reshape=False), 0, 1) for gt_img in gt_imgs])
    return imgs_rot135, gt_imgs_rot135


# def augment_contrast(imgs, gt_imgs):
#     datagen = ImageDataGenerator(brightness_range=[0.5, 2.0])
#     augmented_imgs = []
#     for img in imgs:
#         img_batch = np.expand_dims(img, axis=0)
#         for augmented_img in datagen.flow(img_batch, batch_size=1):
#             augmented_imgs.append(augmented_img[0])
#     return np.array(augmented_imgs), gt_imgs


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
  imgs_45, gt_imgs_45 = rot45(imgs, gt_imgs)
  imgs_135, gt_imgs_135 = rot135(imgs, gt_imgs)
  imgs_contrast, gt_imgs_contrast = augment_contrast(imgs, gt_imgs)
  imgs_contrast2, gt_imgs_contrast2 = augment_contrast(imgs, gt_imgs, 2)
  imgs_flipped, gt_imgs_flipped = fliph(imgs, gt_imgs)
  imgs_flipped2, gt_imgs_flipped2 = flipv(imgs, gt_imgs)
  imgs = np.concatenate((imgs, imgs_90, imgs_180, imgs_270, imgs_45, imgs_135, imgs_flipped, imgs_flipped2, imgs_contrast, imgs_contrast2))
  gt_imgs = np.concatenate((gt_imgs, gt_imgs_90, gt_imgs_180, gt_imgs_270, gt_imgs_45, gt_imgs_135, gt_imgs_flipped, gt_imgs_flipped2, gt_imgs_contrast,gt_imgs_contrast2))
  return imgs, gt_imgs
