import cv2
import numpy as np

from daug.utils import generate_transformations
from os.path import join


# plot the original image, the warped image, and the predicted transform
# applied to the warp image to "invert" the transformation
def plot_samples(Ia, Ib, M, mean, prefix=''):
    assert Ia.shape == Ib.shape, 'shapes must match'

    for i, _ in enumerate(Ia):
        crop = (Ia[i].transpose(1, 2, 0) + mean).astype(np.uint8)
        warp = (Ib[i].transpose(1, 2, 0) + mean).astype(np.uint8)

        theta = M[i].reshape((2, 3))
        trns = cv2.warpAffine(warp, theta, crop.shape[0:2],
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        out = np.hstack((crop, warp, trns))
        cv2.imwrite('%s_%d.png' % (prefix, i), out)


# This is slightly different from https://arxiv.org/abs/1703.05593,
# where the dataset is generated in advance and kept fixed.  Here,
# we generate a new transformation every time an image is sampled.
def prepare_synth_batch(fpaths, mean, params):
    Xa = np.empty((len(fpaths), 3, 227, 227), dtype=np.float32)
    Xb = np.empty((len(fpaths), 3, 227, 227), dtype=np.float32)
    M  = np.empty((len(fpaths), 6), dtype=np.float32)

    for i, fpath in enumerate(fpaths):
        img = cv2.imread(fpath)
        crop, warp, trans = crop_transform(img, params)

        Xa[i] = (crop - mean).astype(np.float32).transpose(2, 0, 1)
        Xb[i] = (warp - mean).astype(np.float32).transpose(2, 0, 1)
        M[i]  = trans[:2].flatten()

    return Xa, Xb, M


def prepare_batch(fpath_pairs, mean):
    Xa = np.empty((len(fpath_pairs), 3, 227, 227), dtype=np.float32)
    Xb = np.empty((len(fpath_pairs), 3, 227, 227), dtype=np.float32)

    for i, (fpath1, fpath2) in enumerate(fpath_pairs):
        im1 = cv2.imread(fpath1)
        im2 = cv2.imread(fpath2)

        im1 = center_crop(im1, 227)
        im2 = center_crop(im2, 227)

        Xa[i] = (im1 - mean).astype(np.float32).transpose(2, 0, 1)
        Xb[i] = (im2 - mean).astype(np.float32).transpose(2, 0, 1)

    return Xa, Xb


def get_batch_idx(N, batch_size):
    num_batches = (N + batch_size - 1) / batch_size

    for i in range(num_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        idx = slice(start, end)

        yield i, idx


def train_val_split(voc_fpath):
    root = join(voc_fpath, 'TrainVal', 'VOCdevkit', 'VOC2011')
    train_fpath = join(root, 'ImageSets', 'Main', 'train.txt')
    with open(train_fpath, 'rb') as f:
        train_fnames = f.readlines()
    train_fpaths = [
        join(root, 'JPEGImages', '%s.jpg' % s.strip()) for s in train_fnames
    ]
    valid_fpath = join(root, 'ImageSets', 'Main', 'val.txt')
    with open(valid_fpath, 'rb') as f:
        valid_fnames = f.readlines()
    valid_fpaths = [
        join(root, 'JPEGImages', '%s.jpg' % s.strip()) for s in valid_fnames
    ]

    return train_fpaths, valid_fpaths


def center_crop(img, length):
    if img.shape[0] < length or img.shape[1] < length:
        top = max(0, int(np.ceil((length - img.shape[0]) / 2.)))
        left = max(0, int(np.ceil((length - img.shape[1]) / 2.)))
        img = cv2.copyMakeBorder(
            img, top, top, left, left, borderType=cv2.BORDER_REFLECT_101
        )

    cv2.imwrite('pad.png', img)

    crop_y = int(np.floor((img.shape[0] - length) / 2.))
    crop_x = int(np.floor((img.shape[1] - length) / 2.))
    crop = img[crop_y:crop_y + length, crop_x:crop_x + length]

    return crop


def crop_transform(img, params):

    # take the center crop of the original image as I_{A}
    crop = center_crop(img, 227)

    M, = generate_transformations(
        1, (img.shape[0], img.shape[1]), **params
    )

    # apply T_{\theta_{GT}} to I_{A} to get I_{B}
    warp = cv2.warpAffine(
        crop.astype(np.float32), M[:2], (crop.shape[1], crop.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    return crop, warp, M
