from glob import glob
import os
from count_cell_number_by_tissue import * 

if __name__ == '__main__':
    glob_and_count_cell_centroids_for_all_samples(
        '../GYTM_breast_Cellular_Features_From_Color_Normalization/', 
        '/data2_image_192_168_10_11/data/data/FromHospital/GuoYaoTongMei-HE-RecurrenceMetastasis-Breast/Cell_Segmentation_From_Tiger_By_LiFeng_From_Color_Normalization/', 
        '../GYTM_breast_Cellular_Features_From_Color_Normalization_Add_Tissue/', )