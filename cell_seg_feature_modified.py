'''
Modified from /data/pipelines/extractcellularfeaturefrompng/cell_seg_feature.py
'''
import numpy as np
from skimage import measure
import json
import glob
import pandas as pd
import os 
import sys

def CellFeatureList(region):
    feature_name_list = ["eccentricity","equivalent_diameter","euler_number","extent","area_filled"
                        ,"inertia_tensor_eigvals_x","inertia_tensor_eigvals_y","major_axis_length","minor_axis_length","orientation",
                        "perimeter","solidity","area","bbox_area","convex_area"]
    feature_values = []
    for feature_name in feature_name_list:
        if feature_name == "inertia_tensor_eigvals_x":
            feature_value = getattr(region,"inertia_tensor_eigvals")
            feature_values.append(feature_value[0])
           
        elif feature_name == "inertia_tensor_eigvals_y":
           feature_value = getattr(region,"inertia_tensor_eigvals")
           feature_values.append(feature_value[1])
        else:
            feature_value = getattr(region,feature_name)
            feature_values.append(feature_value)
    return feature_values

def CalCellFeature(input_path,output_path):
    json_path = glob.glob(input_path + "/*.json")
    for json_file_path in json_path:
        cell_feature_data = []
        cell_type_list = []
        print("running:",json_file_path)
        G_name = os.path.basename(json_file_path)[:-5]
        with open(json_file_path, "r") as file:
            json_data = json.load(file)
        for key, value in json_data["nuc"].items():
            centroid = value['centroid']
            contour_data = value["contour"]
            cell_type = value["type"]
            cell_type_list.append(cell_type)
            # 计算细胞轮廓图像的尺寸
            contour_width = max(point[0] for point in contour_data) + 1
            contour_height = max(point[1] for point in contour_data) + 1
            # 创建与细胞轮廓contour相同形状的二值图像
            contour_image = np.zeros((contour_height, contour_width), dtype=np.uint8)
            for point in contour_data:
                contour_image[point[1], point[0]] = 1
            # 使用measure.regionprops函数计算区域属性
            regions = measure.regionprops(contour_image)
            # 遍历每个区域
            
            for region in regions:
                feature_values = CellFeatureList(region=region)
                cell_feature_data.append(feature_values+[centroid[0], centroid[1]])
        df1 = pd.DataFrame(cell_feature_data,columns=["eccentricity","equivalent_diameter","euler_number","extent","area_filled"
                            ,"inertia_tensor_eigvals_x","inertia_tensor_eigvals_y","major_axis_length","minor_axis_length","orientation",
                            "perimeter","solidity","area","bbox_area","convex_area","centroid_y","centroid_x"])
        
        replace_dict = {0: 'epithelium', 1: 'tumer_cell', 2: 'lymphocytes', 3: 'stroma_cell', 4: 'dead_cell'}
        cell_type_list = [replace_dict[i] if i in replace_dict else i for i in cell_type_list]
        
        df2 = pd.DataFrame({"cell_type":cell_type_list})
        df = df1.join(df2)
        df.to_csv(output_path + "/" + G_name + ".csv",index=False)

if __name__ == "__main__":
    input_path,output_path = sys.argv[1:3]
    Cell_Feature = CalCellFeature(input_path, output_path)
