import cv2
import numpy as np

from daug.utils import generate_transformations
from os.path import join


def prepare_batch(fpaths, mean):
    Xa = np.empty((len(fpaths), 3, 227, 227), dtype=np.float32)
    Xb = np.empty((len(fpaths), 3, 227, 227), dtype=np.float32)
    M  = np.empty((len(fpaths), 6), dtype=np.float32)

    for i, fpath in enumerate(fpaths):
        img = cv2.imread(fpath)
        crop, warp, trans = crop_transform(img)

        Xa[i] = (crop - mean).astype(np.float32).transpose(2, 0, 1)
        Xb[i] = (warp - mean).astype(np.float32).transpose(2, 0, 1)
        M[i]  = trans[:2].flatten()

    return Xa, Xb, M


def get_batch_idx(N, batch_size):
    num_batches = (N + batch_size - 1) / batch_size

    for i in range(num_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        idx = slice(start, end)

        yield i, idx


def train_val_split():
    root = ('/media/hdd/hendrik/datasets/pascal-2011/TrainVal/VOCdevkit/'
            'VOC2011/')
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


def main():
    root = ('/media/hdd/hendrik/datasets/pascal-2011/TrainVal/VOCdevkit/'
            'VOC2011/JPEGImages')

    #fpath = join(root, '2009_004625.jpg')
    fpath = join(root, '2008_000405.jpg')
    img = cv2.imread(fpath)

    #cv2.imwrite('img.png', img)

    crop, warp, M = crop_transform(img)
    out = cv2.warpAffine(
        warp, M[:2], (227, 227), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    )
    #cv2.imwrite('img.png', img)
    cv2.imwrite('out.png', out)
    cv2.imwrite('pad.png', np.hstack((crop, warp)))


def crop_transform(img):
    trans_params = {
        'rotation': (0, 0),
        'offset':   (0, 0),
        'flip':     (False, False),
        'shear':    (0., 0.),
        'stretch':  (1. / 2, 2),
    }

    if img.shape[0] < 227 or img.shape[1] < 227:
        top = max(0, (227 - img.shape[0]) / 2)
        left = max(0, (227 - img.shape[1]) / 2)
        img = cv2.copyMakeBorder(
            img, top, top, left, left, borderType=cv2.BORDER_REFLECT_101
        )

    # take the center crop of the original image as I_{A}
    crop_y = int(np.floor((img.shape[0] - 227) / 2.))
    crop_x = int(np.floor((img.shape[1] - 227) / 2.))
    crop = img[crop_y:crop_y + 227, crop_x:crop_x + 227]

    M, = generate_transformations(
        1, (img.shape[0], img.shape[1]), **trans_params
    )

    # apply T_{\theta_{GT}} to I_{A} to get I_{B}
    warp = cv2.warpAffine(
        crop.astype(np.float32), M[:2], (crop.shape[1], crop.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    return crop, warp, M


if __name__ == '__main__':
    main()
