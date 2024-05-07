from glob import glob
import os
from count_cell_number_by_tissue import calculate_cell_density_for_sample


if __name__ == '__main__':
    import multiprocessing
    pool = multiprocessing.Pool(16)

    dir_output = '/home/wangb/projects/20240311_gytm_breast_recurrence_maojy/TCGA_breast_Cellular_Features_From_Color_Normalization_Add_Tissue_Concat_Densities/'
    os.makedirs(dir_output, exist_ok=True)


    for f in glob('../TCGA_breast_Cellular_Features_From_Color_Normalization_Add_Tissue_Concat/TCGA-*csv.gz'):
        print(f)
        path_output = os.path.join(dir_output, os.path.basename(f) + '.csv.gz')
        if os.path.exists(path_output):
            print('exists')
            continue
        pool.apply_async(
        calculate_cell_density_for_sample, (f, '../TCGA_breast_tissue_pixel_count/', path_output),#, 
        error_callback=print,
        )
        
    pool.close()
    pool.join()
