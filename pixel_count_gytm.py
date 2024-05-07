from count_cell_number_by_tissue import glob_and_count_pixel_for_all_samples
if __name__ == '__main__':
    glob_and_count_pixel_for_all_samples(
                '/data2_image_192_168_10_11/data/data/FromHospital/GuoYaoTongMei-HE-RecurrenceMetastasis-Breast/Cell_Segmentation_From_Tiger_By_LiFeng_From_Color_Normalization/',
                        '../GYTM_breast_tissue_pixel_count/')
