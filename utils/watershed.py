import os

import tifffile as tifi
import cv2
import numpy as np
from skimage.filters import threshold_yen


def watershed(mask):
    tmp = mask.copy()

    tmp[tmp > 0] = 255
    tmp = np.uint8(tmp)

    contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tmp, [contours[0]], -1, 255, -1)

    # open_kernel = np.ones((3, 3))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, open_kernel, iterations=3)
    # tifi.imwrite(file_name.replace('.tif', '_open.tif'), opening)

    sure_bg = cv2.dilate(opening, open_kernel, iterations=3)
    # tifi.imwrite(file_name.replace('.tif', '_bg.tif'), sure_bg)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # tifi.imwrite(file_name.replace('.tif', '_dist.tif'), np.uint8(dist_transform))

    # thr_lst = [0.1, 0.3, 0.5, 0.7]
    thr_lst = np.round(np.arange(1, 10) * 0.1, 1)
    # thr_lst = [0.4]

    cpnt_lst = []
    mtmp_lst = []
    for thr in thr_lst:
        _, sure_fg = cv2.threshold(dist_transform, thr * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        sure_fg = cv2.erode(sure_fg, open_kernel)
        # tifi.imwrite(file_name.replace('.tif', '_fg_{}.tif'.format(thr)), sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)
        unknown = cv2.morphologyEx(unknown, cv2.MORPH_CLOSE, open_kernel, iterations=2)
        sure_fg[unknown > 0] = 0
        # tifi.imwrite(file_name.replace('.tif', '_un_{}.tif'.format(thr)), unknown)

        count, markers = cv2.connectedComponents(sure_fg, connectivity=8)
        cpnt_lst.append(count)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB), markers)

        map_tmp = np.zeros(markers.shape, np.uint8)
        map_tmp[markers == -1] = 255
        cv2.drawContours(map_tmp, [contours[0]], -1, 0, 1)
        mtmp_lst.append(map_tmp)
        # map_tmp = cv2.dilate(map_tmp, open_kernel)

    map_final = np.zeros(mask.shape)
    max_cpnt = max(cpnt_lst)
    for i in range(len(mtmp_lst)):
        if cpnt_lst[i] == max_cpnt:
            map_tmp = mtmp_lst[i]
            stack_tmp = map_tmp / 255 + map_final / 255
            stack_tmp[stack_tmp <= 1] = 0
            if stack_tmp.sum() > 0:
                _, lbl = cv2.connectedComponents(map_tmp, connectivity=8)
                for j in range(1, lbl.max()):
                    if stack_tmp[lbl == j].sum() > 0:
                        map_tmp[lbl == j] = 0

            map_final[map_tmp > 0] = 255

    opening[map_final > 0] = 0
    opening = cv2.erode(opening, np.ones((3, 3)))
    opening = cv2.dilate(opening, open_kernel)
    # tifi.imwrite(file_name.replace('.tif', '_sp.tif'), opening)

    return opening, tmp


if __name__ == '__main__':

    file_name = r'D:\test_data\MonkeyBrain\FP200000287BL_A1\sample\2\5990_3226_pred-1.tif'
    # src_img = tifi.imread(r'D:\test_data\MonkeyBrain\FP200000287BL_A1\sample\5990_3226_src.tif')
    mask_img = tifi.imread(file_name)
    watershed(file_name, mask_img)

