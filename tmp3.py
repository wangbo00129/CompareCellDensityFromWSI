from glob import glob
import os
from cell_distance_utils import concat_dict_shortest_distances_for_each_patch_into_one 

if __name__ == '__main__':
    import multiprocessing
    pool = multiprocessing.Pool(8)

    dir_output = '/home/wangb/projects/20240311_gytm_breast_recurrence_maojy/TCGA_breast_Cellular_Features_From_Color_Normalization_Shortest_Distances_Each_Cell_Type_Pair_Concat_To_WSI/'
    os.makedirs(dir_output, exist_ok=True)

    for f in glob('/home/wangb/projects/20240311_gytm_breast_recurrence_maojy/TCGA_breast_Cellular_Features_From_Color_Normalization_Shortest_Distances_Each_Cell_Type_Pair/*pkl'):
        print(f)
        path_output = os.path.join(dir_output, os.path.basename(f) + '.concat_to_wsi.pkl')
        if os.path.exists(path_output):
            print('exists')
            continue
        pool.apply_async(
        concat_dict_shortest_distances_for_each_patch_into_one, (f, path_output),#, 
        error_callback=print,
        )
        
    pool.close()
    pool.join()
