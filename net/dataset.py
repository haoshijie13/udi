import os
import copy
from glob import glob
import random

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import tifffile as tifi


def ij_auto_contrast(img):
    limit = img.size / 10
    threshold = img.size / 5000

    if img.dtype != 'uint8':
        bit_max = 65536
        hist, _ = np.histogram(ij_16_to_8(img).flatten(), 256, [0, 256])
    else:
        bit_max = 256
        hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    hmin = 0
    hmax = 255
    for i in range(1,255):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmin = i
            break
    for i in range(254, 0, -1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmax = i
            break
    dst = copy.deepcopy(img)
    if hmax > hmin:
        hmax = int(hmax * bit_max / 256)
        hmin = int(hmin * bit_max / 256)
        dst[dst < hmin] = hmin
        dst[dst > hmax] = hmax
        cv2.normalize(dst, dst, 0, bit_max - 1, cv2.NORM_MINMAX)
    return dst


class Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        # self.img_lst = glob(os.path.join(data_path, 'train', '*.tif'))
        self.lbl_lst = glob(os.path.join(data_path, 'label', '*.tif'))

    def augment(self, img, flip_type):
        flip = cv2.flip(img, flip_type)
        return flip

    def __getitem__(self, idx):
        # img_file = self.img_lst[idx]
        # label_file = img_file.replace('train', 'label')
        label_file = self.lbl_lst[idx]
        img_file = label_file.replace('label', 'train')

        # img = cv2.imread(img_file)
        img = tifi.imread(img_file)
        lbl = cv2.imread(label_file)

        # lbl = lbl[:img.shape[0], :img.shape[1]]

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY)

        if lbl.max() > 1:
            lbl = lbl / 255

        # flip_type = random.choice([-1, 0, 1, 2])
        # if flip_type != 2:
        #     img = self.augment(img, flip_type)
        #     lbl = self.augment(lbl, flip_type)

        img = img.reshape(1, img.shape[0], img.shape[1])
        lbl = lbl.reshape(1, lbl.shape[0], lbl.shape[1])

        return img, lbl

    def __len__(self):
        # return len(self.img_lst)
        return len(self.lbl_lst)


def split_img(img_path, step):
    out_path = os.path.join(os.path.dirname(img_path), 'tst')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img_lst = glob(os.path.join(img_path, '*.jpg'))

    for img_file in img_lst:
        img = cv2.imread(img_file)
        h, w = img.shape[:2]

        lbl_file = img_file.replace('img', 'label').replace('.jpg', '_label.tif')
        lbl = cv2.imread(lbl_file)

        rr = int(h / step)
        cc = int(w / step)

        for i in range(rr):
            for j in range(cc):
                ed_r = min((i+1) * step, h)
                ed_c = min((j+1) * step, w)

                if not os.path.exists(os.path.join(out_path, 'img')):
                    os.makedirs(os.path.join(out_path, 'img'))
                cv2.imwrite(os.path.join(out_path, 'img', os.path.basename(img_file).replace('.jpg', '_{}_{}.jpg'.format(i, j))), img[i*step:ed_r, j*step:ed_c])

                if not os.path.exists(os.path.join(out_path, 'lbl')):
                    os.makedirs(os.path.join(out_path, 'lbl'))
                tifi.imwrite(os.path.join(out_path, 'lbl', os.path.basename(lbl_file).replace('_label.tif', '_{}_{}_label.tif'.format(i, j))), lbl[i*step:ed_r, j*step:ed_c])


def split_single(img_path, step=512, overlap=0):
    img = tifi.imread(img_path)
    # img = ij_auto_contrast(img)

    h, w = img.shape
    rr = int(h / step) + 1
    cc = int(w / step) + 1

    for i in range(rr):
        for j in range(cc):
            st_r = max(i * step - overlap * step, 0)
            st_c = max(j * step - overlap * step, 0)

            ed_r = min(st_r + step, h)
            ed_c = min(st_c + step, w)

            # out_path = os.path.join(os.path.dirname(img_path), 'crop')
            out_path = r'D:\test_data\singlecell\RNA_Segment\SS200000158BL_A5\crop'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            if img[st_r:ed_r, st_c:ed_c].mean() <= 2:
                continue

            tifi.imwrite(os.path.join(out_path, 'SS200000158BL_A5_{}_{}.tif').format(st_r, st_c), img[st_r:ed_r, st_c:ed_c])

    return out_path, (h, w)


if __name__ == '__main__':
    # dataset = Loader(r'D:\CODES\yeast\train\data')
    # print(len(dataset))
    # train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    # for img, lbl in train_loader:
    #     print(img.shape)
    img_path = r'D:\test_data\singlecell\RNA_Segment\SS200000158BL_A5\SS200000158BL_A5_rna_conv.tif'
    step = 1024
    split_single(img_path, step)
