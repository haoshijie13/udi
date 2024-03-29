import os
import sys

import cv2
from PIL import Image
import numpy as np
import tifffile as tifi
from skimage.measure import label, regionprops

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from net.feature_vision import predict
from net.predict import predict
# from net.predict_onnx import predict
from img_process import post_process, count_mask, batch_post_process


class RnaSegment(object):
    def __init__(self, img_file, seg_type='s', method=1, gpu=False, version='0418'):
        self.img_file = img_file
        self.out_path = os.path.dirname(img_file)
        self.seg_type = seg_type
        self.method = method
        self.gpu = gpu
        self.version = version

        # self.img = tifi.imread(img_file)

    def segment_dp(self):
        mask = predict(self.img_file, self.out_path, step=512, overlap=0.1, gpu=self.gpu, version=self.version)
        # mask is a numpy array
        # post = post_process(mask)
        post = batch_post_process(mask)
        # tifi.imwrite(self.img_file.replace('.tif', '_mask.tif'), post)
        image = Image.fromarray(post)
        image.save(self.img_file.replace('.tif', '_mask.tif'), compression="tiff_lzw")

        # cell_count = count_mask(post)
        # with open(os.path.join(os.path.dirname(self.img_file), 'cell_count.txt'), 'w') as fw:
        #     fw.write(str(cell_count))

        cnts, _ = cv2.findContours(post, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # with open(os.path.join(os.path.dirname(self.img_file), 'cell_count.txt'), 'w') as fw:
        #     fw.write(str(len(cnts)))

        outline = np.zeros(post.shape, np.uint8)
        cv2.drawContours(outline, cnts, -1, 255, 1)
        # tifi.imwrite(self.img_file.replace('.tif', '_outline.tif'), outline)
        # image = Image.fromarray(outline)
        # image.save(self.img_file.replace('.tif', '_outline.tif'), compression="tiff_lzw")

        return self.img_file.replace('.tif', '_mask.tif')

    def process(self):
        if self.method == 1:
            mask_file = self.segment_dp()

        return mask_file


if __name__ == '__main__':
    img_file = r'D:\test_data\singlecell\gem_data\output_tst\C01533F3_torch\tst\C01533F3-1.tif'
    gpu = True
    version = '0606'
    rs = RnaSegment(img_file, gpu=gpu, version='0606')
    rs.process()


