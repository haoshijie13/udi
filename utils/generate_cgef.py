import os
import glog
from gefpy import cgef_writer_cy, bgef_writer_cy


class CgefGenrate(object):
    def __init__(self, cgef_out_dir):
        """generate cgef from bgef and mask or bgef and ssdna image

        :param cgef_out_dir: the path of the directory to save generated cgef
        """
        self.cgef_out_dir = cgef_out_dir
        # self.cgef_out_parent_path = os.path.dirname(cgef_out_path)
        if not os.path.exists(self.cgef_out_dir):
            os.makedirs(self.cgef_out_dir)

    def get_file_name(self, source_file_path, ext=None):
        ext = ext.lstrip('.') if ext is not None else ""
        file_name = os.path.basename(source_file_path)
        file_prefix = os.path.splitext(file_name)[0]
        if ext == "":
            return file_prefix
        else:
            return f"{file_prefix}.{ext}"

    def generate_bgef(self, gem_path, threads=10):
        file_name = self.get_file_name(gem_path, 'bgef')
        bgef_path = os.path.join(self.cgef_out_dir, file_name)
        glog.info(f"start to generate bgef({bgef_path})")
        if os.path.exists(bgef_path):
            os.remove(bgef_path)
        bgef_writer_cy.generate_bgef(gem_path, bgef_path, n_thread=threads, bin_sizes=[1])
        glog.info(f"generate bgef finished")
        return bgef_path

    def generate(self,
                 gem_path: str = None,
                 mask_path: str = None,
                 ):

        bgef_path = self.generate_bgef(gem_path)
        file_name = self.get_file_name(bgef_path, 'cellbin.gef')
        cgef_out_path = os.path.join(self.cgef_out_dir, file_name)
        cgef_writer_cy.generate_cgef(cgef_out_path, bgef_path, mask_path, [256, 256])
        return cgef_out_path


if __name__ == '__main__':
    # save_p = "/media/Data/dzh/output/singlecell/large_cell_small_cell_result/result/B02704F3/test_new"
    # gem_path = "/media/Data/dzh/data/single_cell/large_cell_and_small_cell/MouseEgg_6weeks/B02704F3.gem.gz"
    # mask_file = "/media/Data/dzh/output/singlecell/large_cell_small_cell_result/result/B02704F3/B02704F3_mask.tif"
    # cg = CgefGenrate(save_p)
    # cg.generate(
    #     gem_path,
    #     mask_file
    # )
    save_p = "/media/Data/dzh/output/singlecell/large_cell_small_cell_result/result/D01767E5/test_new"
    gem_path = "/media/Data/dzh/data/single_cell/large_cell_and_small_cell/MouseEgg_6weeks/D01767E5.gem.gz"
    mask_file = "/media/Data/dzh/output/singlecell/large_cell_small_cell_result/result/D01767E5/D01767E5_mask.tif"
    cg = CgefGenrate(save_p)
    cg.generate(
        gem_path,
        mask_file
    )