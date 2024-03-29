import os
import sys
import time
from glob import glob

import cv2
from PIL import Image
import numpy as np
import torch
import tifffile as tifi
from skimage.measure import label, regionprops
import glog
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .net_model import UNet
# from .pp_model import UNetPP
import pp_model
# import pp_model_5
from utils.img_process import img_conv, ij_auto_contrast


def pred_single(img, device, net):
    # if img.ndim == 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    # img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    pred = net(img_tensor)
    pred = torch.sigmoid(pred)
    pred = np.array(pred.data.cpu()[0])[0]
    mask = pred.copy()
    mask[mask >= 0.5] = 255
    mask[mask < 0.5] = 0
    # pred[pred > 255] = 255
    # pred[pred < 0] = 0
    pred = np.uint8(pred * 255)
    return np.uint8(pred), np.uint8(mask)


def predict(src_file, out_path, step=512, overlap=0.1, gpu=False, version='0418'):
    t0 = time.time()

    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    src_lst = [518, 519, 523, 524, 525, 526, 527, 531, 606]
    double_lst = [524, 525, 526, 527, 531]
    bri_lst = [525, 526, 527, 531, 606]
    five_lst = [523]

    if int(version) < 418:
        net = UNet(n_channels=1, n_classes=1)
    elif int(version) in double_lst:
        net = pp_model.UNetPP(n_channels=2, n_classes=1)
    elif int(version) in five_lst:
        net = pp_model_5.UNetPP(n_channels=1, n_classes=1)
    else:
        net = pp_model.UNetPP(n_channels=1, n_classes=1)

    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'best_model_{}.pth'.format(version)), map_location=device))
    net.eval()
    glog.info(f"Model has been loaded to {device}")

    src = tifi.imread(src_file)
    if int(version) not in src_lst:
        src = img_conv(src)
    elif int(version) in double_lst:
        conv = img_conv(src)
    if int(version) in bri_lst:
        src = ij_auto_contrast(src)
        if int(version) in double_lst:
            conv = ij_auto_contrast(conv)
    tot_img = np.zeros(src.shape, np.uint8)
    overlap_img = np.zeros(src.shape, np.uint8)

    # tifi.imwrite((os.path.join(out_path, os.path.basename(src_file).replace('.tif', '_conv.tif'))), src)

    h, w = src.shape
    rr = int(h / ((1 - overlap) * step)) + 1
    cc = int(w / ((1 - overlap) * step)) + 1

    for i in range(rr):
        for j in range(cc):
            st_r = round(max(i * step - i * overlap * step, 0))
            st_c = round(max(j * step - j * overlap * step, 0))

            ed_r = round(min(st_r + step, h))
            ed_c = round(min(st_c + step, w))

            # if not os.path.exists(os.path.join(out_path, 'crop')):
            #     os.makedirs(os.path.join(out_path, 'crop'))

            img = src[st_r:ed_r, st_c:ed_c]
            if int(version) in double_lst:
                img = np.array([src[st_r:ed_r, st_c:ed_c], conv[st_r:ed_r, st_c:ed_c]])
            if int(version) not in src_lst and img.mean() <= 2:
                tot_img[st_r:ed_r, st_c:ed_c] = 0
                continue
            # tifi.imwrite(os.path.join(out_path, 'crop', '{}_{}.tif'.format(st_r, st_c)), img)

            pred, mask = pred_single(img, device, net)

            # if not os.path.exists(os.path.join(out_path, 'pred')):
            #     os.makedirs(os.path.join(out_path, 'pred'))
            # tifi.imwrite(os.path.join(out_path, 'pred', '{}_{}.tif'.format(st_r, st_c)), pred)
            
            # if not os.path.exists(os.path.join(out_path, 'mask')):
            #     os.makedirs(os.path.join(out_path, 'mask'))
            # tifi.imwrite(os.path.join(out_path, 'mask', '{}_{}.tif'.format(st_r, st_c)), mask)

            tot_img[st_r:ed_r, st_c:ed_c][mask > 0] = 255

            # mkst_r = 0
            # mkst_c = 0
            # mked_r = step
            # mked_c = step
            # if st_r != 0:
            #     st_r += int(overlap * step // 2)
            #     mkst_r = int(overlap * step // 2)
            # if st_c != 0:
            #     st_c += int(overlap * step // 2)
            #     mkst_c = int(overlap * step // 2)
            # if ed_r != h:
            #     ed_r -= int(overlap * step // 2)
            #     mked_r = -int(overlap * step // 2)
            # if ed_c != w:
            #     ed_c -= int(overlap * step // 2)
            #     mked_c = -int(overlap * step // 2)
            # mask = mask[mkst_r:mked_r, mkst_c:mked_c]
            #
            # tot_img[st_r:ed_r, st_c:ed_c] = mask

    # tot_img = post_process(tot_img)

    # tifi.imwrite(src_file.replace('.tif', '_raw_mask.tif'), tot_img)
    image = Image.fromarray(tot_img)
    image.save(src_file.replace('.tif', '_raw_mask.tif'), compression="tiff_lzw")

    t1 = time.time()
    glog.info('Prediction cost: {}'.format(t1 - t0))

    return tot_img


if __name__ == '__main__':
    src_file = r'D:\test_data\singlecell\RNA_Segment\SS200000177BR_C6\SS200000177BR_C6_rna_conv.tif'
    out_path = r'D:\test_data\singlecell\RNA_Segment\SS200000177BR_C6'
    predict(src_file, out_path, 1024)

    # mask_file = r'D:\test_data\singlecell\RNA_Segment\SS200000158BL_A5\SS200000158BL_A5_rna_conv_mask.tif'
    # pred_path = os.path.join(out_path, 'pred')
    # stitch(mask_file, pred_path)

