import os
import sys
import copy
import time
from glob import glob

import cv2
import glog
import numpy as np
import tifffile as tifi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

sys.path.append(os.path.dirname(__file__))
import pit_segment
from watershed import watershed
from wsi_split import SplitWSI
from PIL import Image


def img_conv(img):
    kerl = np.ones((13, 13))
    conv = cv2.filter2D(img, -1, kerl)
    return conv


def contour_interpolate(mask_temp, value=255):
    tmp = mask_temp.copy()
    tmp[tmp == value] = 0

    img = mask_temp.copy()
    img[img != value] = 0
    img[img > 0] = 255
    img = np.uint8(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])

    img = np.uint16(img)
    img[img > 0] = value
    cv2.drawContours(img, [hull], -1, value, thickness=-1)

    img[img + tmp > value] = 0

    return img


def is_border(img, i, j):
    s_r = [-1, 0, 1]
    s_c = [-1, 0, 1]
    if i == 0: s_r = s_r[1:]
    if i == img.shape[0] - 1: s_r = s_r[:-1]
    if j == 0: s_c = s_c[1:]
    if j == img.shape[1] - 1: s_c = s_c[:-1]
    for r in s_r:
        for c in s_c:
            if img[i + r][j + c] != 0 and img[i + r][j + c] != img[i][j]:
                return 1
    return 0


def border_map(img):
    map = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 0:
                continue
            map[i][j] = is_border(img, i, j)
    return map


def f_fusion(img1, img2):
    img1 = cv2.bitwise_or(img1, img2)
    return img1


def batch_post_process(mask, win_size=(15000, 15000)):
    # mask = tifi.imread(mask_path)
    overlap = int(win_size[0] * 0.1)
    sp_run = SplitWSI(img=mask, win_shape=win_size, overlap=overlap, batch_size=1, need_fun_ret=False,
                      need_combine_ret=True,
                      editable=False)
    sp_run.f_set_run_fun(post_process)
    sp_run.f_set_fusion_fun(f_fusion)
    _, _, result_array = sp_run.f_split2run()
    # image = Image.fromarray(result_array)
    # image.save(result_path, compression="tiff_lzw")
    return result_array


def post_process(mask):
    # mask is numpy array (raw mask)
    t0 = time.time()

    # mask = pit_segment.entry(mask)

    label_mask = label(mask, connectivity=2)
    props = regionprops(label_mask, label_mask)

    for idx, obj in enumerate(props):
        if idx == 573:
            print(1)
        bbox = obj['bbox']
        label_mask_temp = label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]].copy()
        if obj['filled_area'] < 80:
            # if obj['filled_area'] < 0:
            label_mask_temp[label_mask_temp == obj['label']] = 0
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = label_mask_temp
        else:
            tmp_mask = label_mask_temp.copy()
            tmp_mask[tmp_mask != obj['label']] = 0

            tmp_mask, tmp_area = watershed(tmp_mask)
            tmp_mask = np.uint32(tmp_mask)
            tmp_mask[tmp_mask > 0] = obj['label']

            label_mask_temp[tmp_area > 0] = tmp_mask[tmp_area > 0]

            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]][tmp_area > 0] = label_mask_temp[tmp_area > 0]

            # intp = contour_interpolate(label_mask_temp, obj['label'])
            # label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]][intp == obj['label']] = obj['label']

    label_mask[label_mask > 0] = 255
    label_mask = label(label_mask, connectivity=2)
    props = regionprops(label_mask, label_mask)

    for idx, obj in enumerate(props):
        bbox = obj['bbox']
        label_mask_temp = label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        if obj['filled_area'] < 80:
            label_mask_temp[label_mask_temp == obj['label']] = 0
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = label_mask_temp
            continue

        # if obj['filled_area'] < 2000:
        if True:
            tmp_mask = label_mask_temp.copy()
            tmp_mask[tmp_mask != obj['label']] = 0
            tmp_mask[tmp_mask > 0] = 255
            tmp_mask = np.uint8(tmp_mask)
            contours, _ = cv2.findContours(tmp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if 0 not in contours[0]:
                if pit_segment.check_shape(contours[0]):
                    label_mask_temp[label_mask_temp == obj['label']] = 0
                    label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = label_mask_temp
                    # continue

        intp = contour_interpolate(label_mask_temp, obj['label'])
        label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]][intp == obj['label']] = obj['label']

    map = border_map(label_mask)

    label_mask[map > 0] = 0

    label_mask[label_mask > 0] = 255

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # label_mask = cv2.morphologyEx(np.uint8(label_mask), cv2.MORPH_OPEN, kernel)

    # label_mask = remove_small_objects(label_mask.astype(np.bool), min_size=10)

    t1 = time.time()
    glog.info('Post process cost: {}'.format(t1 - t0))

    return np.uint8(label_mask)


def count_mask(mask):
    label_mask = label(mask, connectivity=2)
    props = regionprops(label_mask, label_mask)

    return len(props)


def ij_auto_contrast(img):
    limit = img.size / 10
    threshold = img.size / 5000

    if img.dtype != 'uint8':
        bit_max = 65536
    else:
        bit_max = 256
    hist, _ = np.histogram(img.flatten(), bit_max, [0, bit_max])
    hmin = 0
    hmax = bit_max - 1
    for i in range(1, bit_max - 1):
        # for i in range(np.min(img) + 1, np.max(img)):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmin = i
            break
    for i in range(bit_max - 2, 0, -1):
        # for j in range(np.max(img) - 1, 0, -1):
        count = hist[i]
        if count > limit:
            continue
        if count > threshold:
            hmax = i
            break
    dst = copy.deepcopy(img)
    if hmax > hmin:
        # hmin = max(0, hmin - 30)
        # hmax = int(hmax * bit_max / 256)
        # hmin = int(hmin * bit_max / 256)
        dst[dst < hmin] = hmin
        dst[dst > hmax] = hmax
        # cv2.normalize(dst, dst, 0, bit_max - 1, cv2.NORM_MINMAX)
        if bit_max == 256:
            dst = np.uint8((dst - hmin) / (hmax - hmin) * (bit_max - 1))
        elif bit_max == 65536:
            # dst = np.uint16((dst - hmin) / (hmax - hmin) * (bit_max - 1))
            dst = np.uint8((dst - hmin) / (hmax - hmin) * 255)
    return dst


if __name__ == '__main__':
    # # path = r'D:\test_data\singlecell\RNA_Segment\result_0606'
    # # dir_lst = os.listdir(path)
    # # raw_lst = []
    # # for directory in dir_lst:
    # #     raw_file = os.path.join(path, directory, directory + '_raw_mask.tif')
    # #     raw_lst.append(raw_file)
    # raw_file = r'D:\test_data\singlecell\gem_data\output_tst\C01533C4_0707\C01533C4_raw_mask.tif'
    # # raw_file = r'D:\test_data\singlecell\gem_data\output_tst\SS200001410TR_B1_torch\SS200001410TR_B1_raw_mask.tif'
    # # raw_file = r'D:\test_data\MonkeyBrain\FP200000287BL_A1\sample\1\FP200000287BL_A1_raw_mask.tif'
    # raw_lst = [raw_file]
    # for raw_file in raw_lst:
    #     mask_file = raw_file.replace('_raw_mask.tif', '_mask.tif')
    #     # if not os.path.exists(mask_file):
    #     raw_mask = tifi.imread(raw_file)
    #     mask = post_process(raw_mask)
    #     tifi.imwrite(mask_file, mask)
    #
    #     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     with open(os.path.join(os.path.dirname(mask_file), 'cell_count.txt'), 'w') as fw:
    #         fw.write(str(len(cnts)))
    #
    #     outline = np.zeros(mask.shape, np.uint8)
    #     cv2.drawContours(outline, cnts, -1, 255, 1)
    #     tifi.imwrite(mask_file.replace('_mask.tif', '_outline.tif'), outline)
    #
    #     print('Finish {}'.format(raw_file))
    # # raw_mask = tifi.imread(r'D:\test_data\MonkeyBrain\FP200000287BL_A1\FP200000287BL_A1_raw_mask.tif')
    # # tifi.imwrite(r'D:\test_data\MonkeyBrain\FP200000287BL_A1\FP200000287BL_A1_raw_mask-1.tif', post_process(raw_mask))
    mask_path = "/media/Data/dzh/output/singlecell/large_cell_small_cell_result/result/A03099C1/A03099C1_raw_mask.tif"
    result_path = "/media/Data/dzh/output/singlecell/large_cell_small_cell_result/result/A03099C1/A03099C1_mask_batch.tif"
    batch_post_process(mask_path, result_path)
