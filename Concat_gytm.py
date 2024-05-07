from count_cell_number_by_tissue import *

if __name__ == '__main__':
    concat_cell_feature_csvs(
        glob_str = '../GYTM_breast_Cellular_Features_From_Color_Normalization_Add_Tissue/*/',
        dir_count_output = '../GYTM_breast_Cellular_Features_From_Color_Normalization_Add_Tissue_Concat/'
    )
    
