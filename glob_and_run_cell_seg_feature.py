import os
from glob import glob
from cell_seg_feature_modified import CalCellFeature
import multiprocessing

dir_output_parent = '/home/wangb/projects/20240311_gytm_breast_recurrence_maojy/TCGA_breast_Cellular_Features_From_Color_Normalization/'

if __name__ == '__main__':
    pool = multiprocessing.Pool(8)

    for folder_containing_jsons in glob('/data/data/TCGA/breast/Hovernet_Output_From_Color_Normalization/*/json/'):
        dir_output_csv = os.path.join(dir_output_parent, os.path.basename(os.path.dirname(os.path.dirname(folder_containing_jsons))))
        print('input', folder_containing_jsons, 'output', dir_output_csv)
        os.makedirs(dir_output_csv, exist_ok=True)
        pool.apply_async(CalCellFeature, (folder_containing_jsons, dir_output_csv))
    
    pool.close()
    pool.join()

