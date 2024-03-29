import os
import time
import shutil

import glog

from utils.RNASeg_class import RnaSegment
from utils.generate_gene_expression import gem2img, create_gxp
# from utils.fast_v2 import expression_correct
from utils.correction import Fast


def entry(gem_file, out_path, no_exp=False, seg_type='s', method=1, gpu=False, version='0418', correction='10,20,30'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    t0 = time.time()
    glog.info('Read gem file...')

    img_file = os.path.join(out_path, os.path.basename(gem_file).replace('.gem.gz', '.tif'))
    # img_file = os.path.join(out_path, os.path.basename(gem_file).replace('.gem', '.tif'))
    if not os.path.exists(img_file):
        img_file = gem2img(gem_file, out_path)

    t1 = time.time()
    glog.info('Create vision image cost: {}'.format(t1 - t0))

    glog.info('Process segmentation...')

    rs = RnaSegment(img_file, seg_type=seg_type, method=method, gpu=gpu, version=version)
    mask_file = rs.process()

    t2 = time.time()
    glog.info('Segmentation total cost: {}'.format(t2 - t1))

    if not no_exp:
        glog.info('Create cell bin matrix...')

        create_gxp(gem_file, mask_file, out_path)

        t3 = time.time()
        glog.info('Analysis cost: {}'.format(t3 - t2))

        if len(correction) > 1:
            glog.info('Process correction...')

            dist_lst = [int(val) for val in correction.split(',')]
            for dist in dist_lst:
                cr = Fast(mask_file, out_path, dist)
                cr.process()

                corr_file = mask_file.replace('.tif', f'_corr{dist}.tif')
                create_gxp(gem_file, corr_file, out_path, suffix=f'corr{dist}')

            t4 = time.time()
            glog.info('Total correction cost: {}'.format(t4 - t3))

    # shutil.rmtree(os.path.join(out_path, 'crop'))
    # shutil.rmtree(os.path.join(out_path, 'mask'))
    # shutil.rmtree(os.path.join(out_path, 'pred'))

    t4 = time.time()
    glog.info('Total cost: {}'.format(t4 - t0))

    return 0


def main():
    import argparse
    ArgParser = argparse.ArgumentParser()
    ArgParser.add_argument("-i", "--input_file", action="store", required=True, dest='gem_file', default=None, type=str,
                           help="Matrix input.")
    ArgParser.add_argument("-o", "--out_path", action="store", required=True, dest='out_path', default=None, type=str,
                           help="Output path.")
    ArgParser.add_argument("-n", "--no_gene_exp", action="store_true", required=False, dest='gene_exp', default=False,
                           help="Do not creatre gene expression matrix.")
    ArgParser.add_argument("-t", "--seg_type", action="store", required=False, dest='seg_type', default='s', type=str,
                           help="Type of source matrix.")
    ArgParser.add_argument("-m", "--method", action="store", required=False, dest='method', default=1, type=int,
                           help="Seg method. 0:traditional, 1:deeplearning.")
    ArgParser.add_argument("-g", "--gpu", action="store_true", required=False, dest='gpu', default=False,
                           help="Use GPU.")
    ArgParser.add_argument("-v", "--version", action="store", required=False, dest='version', default='0418',
                           help="Model version.")
    ArgParser.add_argument("-c", "--correction", action="store", required=False, dest='correction', default='10,20,30',
                           help="Correction distances.")

    para, _ = ArgParser.parse_known_args()

    entry(para.gem_file, para.out_path, para.gene_exp, para.seg_type, para.method, para.gpu, para.version, para.correction)


if __name__ == '__main__':
    main()



