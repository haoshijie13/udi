#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/8/6 下午1:35
# @Author : ChaoZhang
# @Version：V 0.1
# @File : generate_gene_expression.py
# @Emial : zhangchao5@genomics.cn
import cv2
import os
import gzip
import time

import numpy as np
import pandas as pd
import tifffile as tifi

from PIL import Image
from warnings import filterwarnings

filterwarnings("ignore")


def parse_head(gem):
    """parse the header info"""
    if gem.endswith(".gz"):
        f = gzip.open(gem, "rb")
    else:
        f = open(gem, "rb")

    header = ""
    num_of_head_lines = 0
    eoh = 0

    for i, l in enumerate(f):
        l = l.decode("utf-8")
        if l.startswith("#"):
            header += l
            num_of_head_lines += 1
            eoh = f.tell()  # get end of header position
        else:
            break
    f.seek(eoh)

    return f, num_of_head_lines, header


def create_gxp(gem_file, mask_file, out_path, suffix=None):
    os.makedirs(out_path, exist_ok=True)

    print("Loading mask file...")
    # correct by tissue mask
    # im = tifi.imread(mask_file)
    # tissue = np.ones(im.shape, np.uint8)
    # height, width = im.shape
    # kernel = np.ones((100, 100), np.uint8)
    # erode = cv2.erode(tissue, kernel, iterations=1)
    # tissue_mask = Image.fromarray(erode)
    # tm = tissue_mask.resize((width, height), Image.NEAREST)
    # tm1 = np.array(tm)
    # mask = tm1 * im
    # mask = mask.T
    mask = tifi.imread(mask_file)
    # im = mask
    num1, maskImg = cv2.connectedComponents(mask)

    # sample pseudo cell from background
    # unique, counts = np.unique(maskImg, return_counts=True)
    # area = pd.DataFrame(zip(unique, counts))
    # area = area[area[0] != 0]

    # pseudo_cell_area = area[1].median()
    # pseudo_bg_cell = np.zeros([*im.shape, 3])
    # pseudo_bg_cell = np.zeros(im.shape)
    # for i in range(0, im.shape[0], 100):
    #     for j in range(0, im.shape[1], 100):
    #         pseudo_bg_cell = cv2.circle(pseudo_bg_cell, (i, j), int(np.ceil(np.sqrt(pseudo_cell_area / np.pi))), 100,
    #                                     -1)
    # pseudo_bg_cell = (pseudo_bg_cell[..., 0] > 0).astype(np.uint8)
    # pseudo_bg_cell = (1 - tm1) * pseudo_bg_cell
    # pseudo_bg_cell = pseudo_bg_cell.T

    # _, pseudo_maskImg = cv2.connectedComponents(pseudo_bg_cell)

    # tifi.imwrite(os.path.join(outpath, "T33_mask_4.14.tif"), maskImg)

    # rotImg = np.rot90(maskImg)
    # maskImg = cv2.flip(rotImg, 0)
    print("Reading data..")
    typeColumn = {"geneID": 'str', "x": np.uint32, "y": np.uint32, "values": np.uint32, "UMICount": np.uint32}

    try:
        _, header, _ = parse_head(gem_file)
        genedf = pd.read_csv(gem_file, header=header, sep='\t', dtype=typeColumn)
    except:
        print("Error: there are error in loading gene file, please check the file's header.")

    tissuedf = pd.DataFrame()
    # pseudo_tissuedf = pd.DataFrame()

    dst = np.nonzero(maskImg)
    dst2 = np.nonzero(mask)
    # pseudo_dst = np.nonzero(pseudo_maskImg)

    print("Dumping results...")
    tissuedf['x'] = dst[1] + genedf['x'].min()
    tissuedf['y'] = dst[0] + genedf['y'].min()
    # tissuedf['x'] = dst[1]
    # tissuedf['y'] = dst[0]
    tissuedf['label'] = maskImg[dst]
    cell_area_info = tissuedf.groupby('label').count()
    cell_area_info = cell_area_info.drop(columns=['y'])
    cell_area_info = cell_area_info.rename(columns={'x': 'cell_area'})

    area_name = "cell_area.txt"
    if suffix is not None:
        area_name = area_name.replace('.txt', f'_{suffix}.txt')
    cell_area_info.to_csv(os.path.join(out_path, area_name), sep='\t')

    # pseudo_tissuedf['x'] = pseudo_dst[1]  # + 25545
    # pseudo_tissuedf['y'] = pseudo_dst[0]  # + 13994
    # pseudo_tissuedf['label'] = pseudo_maskImg[pseudo_dst]

    res = pd.merge(genedf, tissuedf, on=['x', 'y'], how='left').fillna(0)
    # bg_res = pd.merge(res[res['label'] == 0][genedf.columns], pseudo_tissuedf, on=['x', 'y'], how='inner')

    out_name = "Cell_GetExp_gene_with_background.txt"
    if suffix is not None:
        out_name = out_name.replace('.txt', f'_{suffix}.txt')
    res.to_csv(os.path.join(out_path, out_name), sep='\t', index=False)
    # bg_res.to_csv(os.path.join(out_path, "Background_Cell_GetExp_gene.txt"), sep='\t', index=False)

    return os.path.join(out_path, out_name)


def gem2img(gem_file, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Read from gem_file
    t0 = time.time()
    f, num_of_header_lines, header = parse_head(gem_file)
    print("Number of header lines: {}".format(num_of_header_lines))
    print("Header info: \n{}".format(header))
    df = pd.read_csv(f, sep='\t', header=0)
    t1 = time.time()
    print("read gem_file: {:.2f}".format(t1-t0))
    # print(df)

    # Get image dimension and count each pixel's gene numbers
    print("min x: {} min y: {}".format(df['x'].min(), df['y'].min()))
    xmi = df['x'].min()
    ymi = df['y'].min()
    # xmi = 0
    # ymi = 2
    df['x'] = df['x']-xmi
    df['y'] = df['y']-ymi

    xma = df['x'].max()
    yma = df['y'].max()
    max_x = xma+1
    max_y = yma+1
    # max_x = 26460
    # max_y = 26457

    # with open(os.path.join(out_path, 'min_max.txt'), 'w') as fw:
    #     fw.write(str(xmi) + '\n')
    #     fw.write(str(ymi) + '\n')
    #     fw.write(str(xma) + '\n')
    #     fw.write(str(yma) + '\n')
    print("image dimension: {} x {} (width x height)".format(max_x, max_y))
    t2 = time.time()
    print("get min max: {:.2f}".format(t2-t1))

    try:
        new_df = df.groupby(['x', 'y']).agg(UMI_sum=('UMICount', 'sum')).reset_index()
    except:
        try:
            new_df = df.groupby(['x', 'y']).agg(UMI_sum=('MIDCount', 'sum')).reset_index()
        except:
            new_df = df.groupby(['x', 'y']).agg(UMI_sum=('MIDCounts', 'sum')).reset_index()
    t3 = time.time()
    print("add up: {:.2f}".format(t3-t2))

    # Set image pixel to gene counts
    if new_df['UMI_sum'].max() < 256:
        image = np.zeros(shape=(max_y, max_x), dtype=np.uint8)
    else:
        image = np.zeros(shape=(max_y, max_x), dtype=np.uint16)
    image[new_df['y'], new_df['x']] = new_df['UMI_sum']
    t4 = time.time()
    print("add up 2: {:.2f}".format(t4-t3))
    # print(new_df)

    # Save image (thumbnail image & crop image) to file
    # filename = os.path.splitext(os.path.basename(gem_file))[0]
    filename = os.path.basename(gem_file).split('.')[0]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img = Image.fromarray(image)
    img.save(os.path.join(out_path, filename+'.tif'), compression="tiff_lzw")
    # tifi.imwrite(os.path.join(out_path, filename+'.tif'), image)

    return os.path.join(out_path, filename + '.tif')


if __name__ == '__main__':
    # gem2img(r'D:\Downloads\C01742C1.gem.gz', r'D:\Downloads')
    # create_gxp(r'D:\Downloads\singlecell\corr_tst\C01527A4.gem.gz', r'D:\Downloads\singlecell\corr_tst\C01527A4\corr_30\C01527A4_mask_edm_30.tif', r'D:\Downloads\singlecell\corr_tst\C01527A4\corr_30')
    gem_file = "/media/Data/dzh/data/single_cell/large_cell_and_small_cell/MouseEgg_6weeks/B02704F3.gem.gz"
    mask_file = "/media/Data/dzh/output/singlecell/large_cell_small_cell_result/result/B02704F3/B02704F3_mask.tif"
    out_path = "/media/Data/dzh/output/singlecell/large_cell_small_cell_result/result/B02704F3/test_new"
    # create_gxp(
    #     gem_file,
    #     mask_file,
    #     out_path,
    # )
    gem2img(gem_file, out_path,)