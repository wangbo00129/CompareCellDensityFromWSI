import cv2
import os
import pandas as pd
from glob import glob
from collections import Counter
import multiprocessing

def count_pixel_in_mask(path_tissue_mask):
    seg = cv2.imread(path_tissue_mask, cv2.IMREAD_GRAYSCALE)
    pixels_each_tissue = Counter(seg.flatten())
    return pixels_each_tissue

def count_cell_number_by_tissue_on_patch(path_csv_from_hovernet, path_tissue_mask, path_output=None):
    '''
    path_csv_from_hovernet, path_tissue_mask should be from the same patch
    path_csv_from_hovernet should contain centroid_x and centroid_y

    path_output: a new csv adding the column recording which tissue the centroid belongs to.
    '''
    df_cell_csv = pd.read_csv(path_csv_from_hovernet)
    df_cell_csv['centroid_x'] = df_cell_csv['centroid_x'].astype(int)
    df_cell_csv['centroid_y'] = df_cell_csv['centroid_y'].astype(int)

    seg = cv2.imread(path_tissue_mask, cv2.IMREAD_GRAYSCALE)

    df_cell_csv.loc[:,'tissue_int'] = seg[df_cell_csv['centroid_x'], df_cell_csv['centroid_y']]

    if path_output:
        df_cell_csv.to_csv(path_output)
    
    return df_cell_csv


def glob_and_count_pixel(folder_containing_tissue_mask_png, path_output):
    
    list_pixel_count_per_tissue_for_all_images = []
	
    list_path_tissue_mask = glob(folder_containing_tissue_mask_png+'/*png')
    
    for path_tissue_mask in list_path_tissue_mask:
        pixel_count_per_tissue = count_pixel_in_mask(path_tissue_mask)
        list_pixel_count_per_tissue_for_all_images.append(pixel_count_per_tissue)

    df_pixel_count = pd.DataFrame(list_pixel_count_per_tissue_for_all_images, index=list_path_tissue_mask)
    df_pixel_count.to_csv(path_output)

def glob_and_count_pixel_for_all_samples(folder_of_folder_containing_tissue_mask_png, folder_parent_output):
    os.makedirs(folder_parent_output, exist_ok=True)
    pool = multiprocessing.Pool(8)
    for folder in glob(folder_of_folder_containing_tissue_mask_png+'/*/'):
        pool.apply_async(
            glob_and_count_pixel,
            (folder, os.path.join(folder_parent_output, os.path.basename(folder.rstrip('/')))+'.pixel_count.csv')
        )
    pool.close()
    pool.join()

def glob_and_count_cell_centroids(folder_containing_csv, folder_containing_tissue_mask_png, folder_output):
    '''
    folder_containing_csv, folder_containing_tissue_mask_png are for one sample
    folder_output: output csvs containing tissues those centroids belongs to.
    '''
    os.makedirs(folder_output, exist_ok=True)
    cell_csvs = glob(folder_containing_csv+'/*csv')
    for csv in cell_csvs:
        path_tissue_mask = os.path.join(folder_containing_tissue_mask_png, os.path.basename(csv).replace('.csv','.png'))
        if not os.path.exists(path_tissue_mask):
            print('{} not exists'.format(path_tissue_mask))
            continue
        count_cell_number_by_tissue_on_patch(csv, path_tissue_mask, os.path.join(folder_output, os.path.basename(csv)))

def glob_and_count_cell_centroids_for_all_samples(folder_of_folder_containing_csv, folder_of_folder_containing_tissue_mask_png, folder_parent_output):
    '''
    to be done
    '''
    pool = multiprocessing.Pool(8)
    folder_samples_containing_csv = glob(folder_of_folder_containing_csv+'/*/')
    for folder_containing_csv in folder_samples_containing_csv:
        folder_containing_tissue_mask_png = os.path.join(folder_of_folder_containing_tissue_mask_png, os.path.basename(folder_containing_csv.rstrip('/')))
        folder_output = os.path.join(folder_parent_output, os.path.basename(folder_containing_csv.rstrip('/')))
        print(folder_containing_csv, folder_containing_tissue_mask_png, folder_output)
        pool.apply_async(
            glob_and_count_cell_centroids, (folder_containing_csv, folder_containing_tissue_mask_png, folder_output))
    pool.close()
    pool.join()
    

def glob_csv_and_concat(glob_str):
    dfs = []
    for p in glob(glob_str):
        df = pd.read_csv(p)
        if not df.empty:
            df.loc[:,'from'] = p
        dfs.append(df)
    df_concat = pd.concat(dfs)
    return df_concat

def concat_cell_feature_csvs_for_one_sample(sample, dir_count_output):
    
    path_output = os.path.join(dir_count_output,
                        os.path.basename(sample.rstrip('/')) + '.csv.gz')

    if os.path.exists(path_output):
        print('{} exists'.format(path_output))
        return

    df_concat = glob_csv_and_concat(sample+'/*csv')
    if df_concat.empty:
        return
    print(sample)
    print(df_concat.shape)
    df_concat.loc[:,'from'] = df_concat.loc[:,'from'].map(os.path.basename)

    df_concat.to_csv(path_output)
    print(df_concat['tissue_int'].value_counts())
    print(df_concat[['cell_type','tissue_int']].value_counts())

def concat_cell_feature_csvs(
    glob_str = '../TCGA_breast_Cellular_Features_From_Color_Normalization_Add_Tissue/*/',
    dir_count_output = '../TCGA_breast_Cellular_Features_From_Color_Normalization_Add_Tissue_Concat/'
    ):
    
    os.makedirs(dir_count_output, exist_ok=True)

    for sample in glob(glob_str): # [:10]    : # !!!!!!
        
        try:
            concat_cell_feature_csvs_for_one_sample(sample, dir_count_output)
        except Exception as e:
            print('{} failed due to {}'.format(sample, e))


def calculate_cell_density_for_sample(path_concat, folder_tissue_pixel_count_for_all_samples = '../TCGA_breast_tissue_pixel_count/', path_output=None):
    '''
    path_concat: concat csv for hovernet output adding tissue_int column. 
    '''
    df_concat = pd.read_csv(path_concat)
    sample_name = os.path.basename(path_concat).replace('.xml.csv.gz', '')
    path_tissue_pixel_count = os.path.join(folder_tissue_pixel_count_for_all_samples, sample_name + '.xml.pixel_count.csv')
    df_tissue_pixel_count = pd.read_csv(path_tissue_pixel_count, index_col=0)
    df_tissue_pixel_count.columns = df_tissue_pixel_count.columns.map(int)
    dict_tissue_pixel_count_wsi = df_tissue_pixel_count.sum(axis=0).to_dict()
#     print(df_tissue_pixel_count)
#     print(df_concat['tissue_int'].value_counts())
#     print(df_concat[['cell_type','tissue_int']].value_counts())
    
    dict_densities_per_tissue = {}
    
    dict_densities_per_tissue['all'] = df_concat['cell_type'].value_counts() / sum(dict_tissue_pixel_count_wsi.values())
    
    for tissue_int, pixel_num in dict_tissue_pixel_count_wsi.items():
#         print(df_concat)
        df_centroids_on_this_tissue_type = df_concat[df_concat['tissue_int'] == tissue_int]
#         print(df_centroids_on_this_tissue_type)
        df_cell_density_this_tissue_type = df_centroids_on_this_tissue_type['cell_type'].value_counts() / pixel_num
#         print('df_cell_density_this_tissue_type')
#         print('tissue_int', tissue_int, pixel_num)
#         print(df_cell_density_this_tissue_type)
        dict_densities_per_tissue[tissue_int] = df_cell_density_this_tissue_type
    
    df_density_per_tissue =  pd.DataFrame(dict_densities_per_tissue)
    df_density_per_tissue.loc[:,'sample'] = sample_name
    
    if path_output:
        df_density_per_tissue.to_csv(path_output)
        
    return df_density_per_tissue

def calculate_cell_percentage_in_each_tissue(path_concat, folder_tissue_pixel_count_for_all_samples = '../TCGA_breast_tissue_pixel_count/'):
    '''
    path_concat: concat csv for hovernet output adding tissue_int column. 
    '''
    df_concat = pd.read_csv(path_concat)

    dict_percentage = {}

    for tissue, df_this_tissue in df_concat.groupby('tissue_int'):
        dict_percentage[tissue] = (df_this_tissue['cell_type'].value_counts() / df_this_tissue.shape[0]).to_dict()
    
    df_percentage = pd.DataFrame(dict_percentage).fillna(0)
    return df_percentage

if __name__ == '__main__':
    # glob_and_count_cell_centroids_for_all_samples(
    #     '../TCGA_breast_Cellular_Features_From_Color_Normalization/', 
    #     '/data/data/TCGA/breast/Cell_Segmentation_From_Tiger_By_LiFeng_From_Color_Normalization/',
    #     '../TCGA_breast_Cellular_Features_From_Color_Normalization_Add_Tissue/', 
    #     )
    

    glob_and_count_pixel_for_all_samples(
        '/data/data/TCGA/breast/Cell_Segmentation_From_Tiger_By_LiFeng_From_Color_Normalization/',
        '../TCGA_breast_tissue_pixel_count/')