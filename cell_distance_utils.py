import math
import pandas as pd
import numpy as np
from itertools import product
import joblib
from glob import glob
import os

def closest_points_for_each(array1, array2):    
    """
    计算array1中的每个点到array2中任一点的最近距离。

    参数:
    array1 (np.array): 形状为(M, 2)的NumPy数组，代表M个点的坐标。
    array2 (np.array): 形状为(N, 2)的NumPy数组，代表N个点的坐标。

    返回:
    np.array: 形状为(M,)的NumPy数组，包含array1中每个点到array2中最近点的距离。
    """
    # print(array1.shape)

    # print(array2.shape)

    # 使用广播机制计算array1中每个点与array2中每个点之间的距离的平方
    squared_distances = np.sum((array1[:, np.newaxis, :] - array2) ** 2, axis=2)
    # print(squared_distances.shape)
    # print(squared_distances)
    
    # 找到每个点的最小距离的索引
    min_indices = np.argmin(squared_distances, axis=1)
    # print(min_indices.shape)
    
    # 使用这些索引找到对应的最近距离
    min_distances = np.sqrt(squared_distances[np.arange(len(array1)), min_indices])
    
    return min_distances

def calculate_distance_between_specific_pair(df_in, cell_type1, cell_type2):
    '''
    Find the shortest distance to cell_type2 for each cell_type1.
    '''
    array1 = df_in[df_in['cell_type'] == cell_type1][['centroid_x', 'centroid_y']].values
    array2 = df_in[df_in['cell_type'] == cell_type2][['centroid_x', 'centroid_y']].values

    shortest_distances = closest_points_for_each(array1, array2)
    return shortest_distances, array1.shape[0], array2.shape[0]
    

def calculate_cell_distance_between_different_kinds_one_patch(df_hovernet_this_patch):
    dict_pair_shortest_distances = {}

    for pair in product(df_hovernet_this_patch['cell_type'].unique(), df_hovernet_this_patch['cell_type'].unique()):
        shortest_distances, num_pair1, num_pair2 = calculate_distance_between_specific_pair(df_hovernet_this_patch, pair[0], pair[1])
        dict_pair_shortest_distances[pair] = shortest_distances
        dict_pair_shortest_distances[pair[0]] = num_pair1
        dict_pair_shortest_distances[pair[1]] = num_pair2

    return dict_pair_shortest_distances

def calculate_cell_distance_between_different_kinds(path_hovernet, path_output=None):
    '''
    path_hovernet should be csv containing centroid_x and centroid_y, cell_type, tissue_int for a whole WSI. 
    Besides, it should contain a column `from` to indicate which patch this cell is from.
    '''
    nested_dict_patch_pairs_distances = {}
    df_hovernet_wsi = pd.read_csv(path_hovernet)
    for patch_name, df_hovernet_this_patch in df_hovernet_wsi.groupby('from'):
        print(patch_name)
        
        
        # df_hovernet_this_patch.to_csv('df_hovernet_this_patch.csv')



        dict_pair_shortest_distances = calculate_cell_distance_between_different_kinds_one_patch(df_hovernet_this_patch)
        # print(dict_pair_shortest_distances)
        # break
        nested_dict_patch_pairs_distances[patch_name] = dict_pair_shortest_distances

    if path_output:
        joblib.dump(nested_dict_patch_pairs_distances, path_output)
    return nested_dict_patch_pairs_distances
    
def concat_dict_shortest_distances_for_each_patch_into_one(path_dict_shortest_distances_for_each_patch, path_output=None):
    '''
    Each key is a patch.
    Each value contains:
        keys:
            (cell_type1, cell_type2)
            or cell_type

        values:
            shortest distances
            or cell number
    '''
    dict_shortest_distances_for_each_patch = joblib.load(path_dict_shortest_distances_for_each_patch)
    
    patches = list(dict_shortest_distances_for_each_patch.keys())
    
    dict_to_return = {}
    
    for p in patches:
        for k,v in dict_shortest_distances_for_each_patch[p].items():
            if isinstance(k, tuple):
                dict_to_return[k] = np.concatenate([dict_to_return.get(k, np.array([])), v])
            elif isinstance(k, str):
                dict_to_return[k] = dict_to_return.get(k, 0) + v
            else:
                raise Exception('not recognized key type')

    if path_output:
        joblib.dump(dict_to_return, path_output)
    return dict_to_return

def calculate_effective_score(dict_shortest_distances_for_each_wsi, threshold=40):
    '''
    threshold unit: pixel
    '''
    dict_effective_score = {}

    for k in dict_shortest_distances_for_each_wsi:
        if isinstance(k, tuple):
            dict_effective_score[k] = (dict_shortest_distances_for_each_wsi[k] < threshold).sum() / dict_shortest_distances_for_each_wsi[k[1]]

    return dict_effective_score

if __name__ == '__main__':
    import multiprocessing
    pool = multiprocessing.Pool(16)

    dir_output = '/home/wangb/projects/20240311_gytm_breast_recurrence_maojy/TCGA_breast_Cellular_Features_From_Color_Normalization_Shortest_Distances_Each_Cell_Type_Pair/'
    os.makedirs(dir_output, exist_ok=True)

    for f in glob('/home/wangb/projects/20240311_gytm_breast_recurrence_maojy/TCGA_breast_Cellular_Features_From_Color_Normalization_Add_Tissue_Concat/*csv.gz'):
        pool.apply_async(
            calculate_cell_distance_between_different_kinds, 
            (f, os.path.join(dir_output, os.path.basename(f) + '.pkl')),
        )
        
    pool.close()
    pool.join()