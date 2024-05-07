import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt 
from itertools import combinations

import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc

# 定义组织类型映射字典
DICT_TISSUE_INT_TO_TISSUE_TYPE = {
    0: 'exclude',
    1: 'invasive tumor',
    2: 'tumor-associated stroma',
    3: 'in-situ tumor',
    4: 'healthy glands',
    5: 'necrosis not in-situ',
    6: 'inflamed stroma',
    7: 'rest'
}

# 定义颜色映射字典，为每种组织类型分配不同的颜色
DICT_TISSUE_TYPE_TO_COLOR = {
    'exclude': [0, 0, 0],  # 黑色
    'invasive tumor': [255, 0, 0],  # 红色
    'tumor-associated stroma': [0, 255, 0],  # 绿色
    'in-situ tumor': [0, 0, 255],  # 蓝色
    'healthy glands': [255, 255, 0],  # 黄色
    'necrosis not in-situ': [255, 0, 255],  # 紫色
    'inflamed stroma': [0, 255, 255],  # 青色
    'rest': [128, 128, 128]  # 灰色
}

def overlay_segmentation(raw_image, seg_image, alpha=0.5):
    """
    将分割图像叠加到原始图像上，并根据颜色值标记组织类型，具有一定透明度。

    参数:
    - raw_image: 原始图像，应为三通道图像。
    - seg_image: 分割图像，一通道图像，颜色值从0到7。
    - alpha: 叠加图像的透明度，范围从0到1。

    返回:
    - overlaid_image: 叠加后的图像。
    """
    # 确保原始图像是三通道的
    if len(raw_image.shape) != 3 or raw_image.shape[2] != 3:
        raise ValueError("原始图像必须是三通道图像。")

    # 确保分割图像是单通道的
    if len(seg_image.shape) != 2 or raw_image.shape[:2] != seg_image.shape:
        raise ValueError("分割图像必须是与原始图像同尺寸的单通道图像。")

    # 初始化叠加后的图像
    overlaid_image = raw_image.copy()

    # 定义组织类型映射字典
    DICT_TISSUE_INT_TO_COLOR = {
        k: DICT_TISSUE_TYPE_TO_COLOR[DICT_TISSUE_INT_TO_TISSUE_TYPE[k]] for k in DICT_TISSUE_INT_TO_TISSUE_TYPE
    }

    # 遍历分割图像的每个像素
    for tissue_type, color_value in DICT_TISSUE_INT_TO_COLOR.items():
        # 检查分割图像中是否存在当前组织类型
        mask = np.where(seg_image == tissue_type)
        
        # 计算叠加颜色
        combined_color = (np.array(color_value) * alpha) + (np.array(raw_image[mask]) * (1 - alpha))
        
        # 将结合后的颜色应用到原始图像的对应位置
        overlaid_image[mask] = combined_color.astype(np.uint8)

    return overlaid_image

def draw_legend(
    dictionary_tissue_int=DICT_TISSUE_INT_TO_TISSUE_TYPE, 
    dictionary_tissue_color=DICT_TISSUE_TYPE_TO_COLOR, legend_title="Legend", alpha=0.5):
    """
    根据组织类型的整数标识和颜色映射绘制半透明图例。

    参数:
    - dictionary_tissue_int: 组织类型的整数标识字典。
    - dictionary_tissue_color: 组织类型的颜色映射字典。
    - legend_title: 图例的标题，默认为 "Legend"。
    - alpha: 图例中彩色方块的透明度。

    返回:
    - legend_image: 包含图例的图像。
    """

    # 计算图例中每个条目的宽度和高度
    legend_entry_width = 400
    legend_entry_height = 25
    legend_entry_spacing = 10

    # 计算图例所需的总高度和宽度
    total_entries = len(dictionary_tissue_int)
    total_height = (total_entries * (legend_entry_height + legend_entry_spacing)) - legend_entry_spacing
    total_width = legend_entry_width

    # 创建一个空白的图例图像
    legend_image = np.zeros((int(total_height), int(total_width), 3), dtype=np.uint8)
    legend_image[:] = (255, 255, 255)  # 设置背景颜色为白色

    # 设置字体和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 0, 0)  # 黑色

    # 初始化y坐标以绘制图例条目
    y_coord = 10

    # 遍历组织类型的整数标识和颜色映射字典，绘制每个图例条目
    for tissue_int, tissue_type in dictionary_tissue_int.items():
        # 将颜色值的每个通道乘以alpha，然后转换为0-255范围
        transparent_color = tuple(np.array(dictionary_tissue_color[tissue_type]) * alpha * 255)
        
        # 绘制半透明矩形
        legend_image = cv2.rectangle(
            legend_image,
            (0, y_coord),
            (legend_entry_width, y_coord + legend_entry_height),
            color=transparent_color,
            thickness=cv2.FILLED
        )
        cv2.putText(
            legend_image,
            tissue_type,
            (10, y_coord + legend_entry_height - 5),
            font,
            font_scale,
            font_color,
            2
        )
        y_coord += legend_entry_height + legend_entry_spacing

    # 添加图例标题
    cv2.putText(
        legend_image,
        legend_title,
        (10, 10),
        font,
        font_scale,
        font_color,
        2
    )

    return legend_image

# # 读取原始图像和分割图像
# raw = cv2.imread('/path/to/raw/image')
# seg = cv2.imread('/path/to/seg/image', cv2.IMREAD_GRAYSCALE)

# # 调用函数进行叠加
# overlaid_image = overlay_segmentation(raw, seg)

# # 显示叠加后的图像
# cv2.imshow('Overlayed Image', overlaid_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def stacked_bar_chart(df):

    # #createDataFrame
    # df = pd. DataFrame ({'Day': ['Mon', 'Tue', 'Wed', 'Thur', 'Fri'],
    #                 'Morning': [44, 46, 49, 59, 54],
    #                 'Evening': [33, 46, 50, 49, 60]})

    #set seaborn plotting aesthetics
    sns.set(style='white')
    #create stacked bar chart
    df.set_index('Day').plot(kind='bar', stacked= True, color=['steelblue', 'red'])
    return 

def boxplot_with_p_value(df, x, y, ax):
    

    order = sorted(df[x].unique())

    ax = sns.boxplot(data=df, x=x, y=y, order=order, ax=ax)

    pairs= list(combinations(order, 2))
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
    annotator.configure(test='Mann-Whitney', text_format='star',line_height=0.03,line_width=1)
    annotator.apply_and_annotate()

    ax.tick_params(which='major',direction='in',length=3,width=1.,labelsize=14,bottom=False)
    for spine in ["top","left","right"]:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(axis='y',ls='--',c='gray')
    ax.set_axisbelow(True)

def plot_roc(label, pred):
    # 绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(label, pred)
    # 计算AUC值
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    ax = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return ax

if __name__ == '__main__':
    df = pd.DataFrame({'day':['Sun', 'Thur', 'Fri', 'Sat'], 'total_bill':[1,2,3,4]}) # sns.load_dataset("tips")

    x = "day"
    y = "total_bill"
    order = ['Sun', 'Thur', 'Fri', 'Sat']
    # fig,ax = plt.subplots(figsize=(5,4),dpi=100,facecolor="w")

    boxplot_with_p_value(df, x, y)
