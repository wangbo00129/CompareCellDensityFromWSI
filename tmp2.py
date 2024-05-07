from count_cell_number_by_tissue import glob_and_count_pixel_for_all_samples
if __name__ == '__main__':
    glob_and_count_pixel_for_all_samples(
                '/data/data/TCGA/breast/Cell_Segmentation_From_Tiger_By_LiFeng_From_Color_Normalization/',
                        '../TCGA_breast_tissue_pixel_count/')
