# target data format
import pandas as pd
import os
import os.path as osp
import json

# filepath = '/data/dataspace/clevr_cogen_a_train/data/train-00000-of-00027.parquet'
filepath = '/data/dataspace/slow_percept-sub1-ori_form-rl/train-00001-of-00058.parquet'
# 读取 parquet 文件
df = pd.read_parquet(filepath)

# 显示前几行数据
print(df.head())

# 查看列信息
print(df.info())

# 显示数据概览
print(df.describe())

import ipdb;ipdb.set_trace()


# # source data format
# root = 'dataset/jihe_train_data/sub1/jsons'
# jsonfiles = os.listdir(root)
# root = 'dataset/jihe_train_data/sub2/jsons'
# jsonfiles2 = os.listdir(root)
# import ipdb;ipdb.set_trace()

# filepath = osp.join(root, '9951.json')
# with open(filepath, 'r') as f:
#     data = json.load(f)

# def coor2txt(coordinates):
#     lines = []
#     circles = []

#     for item in coordinates:
#         # 处理 Line 部分
#         if "Line" in item:
#             for line in item["Line"]:
#                 x1, y1, x2, y2 = line  # 拆分四元组
#                 lines.append(f"({x1}, {y1}) -- ({x2}, {y2})")

#         # 处理 Circle 部分
#         if "Circle" in item:
#             for circle in item["Circle"]:
#                 cx, cy, r = circle  # 提取圆的中心和半径
#                 circles.append(f"({cx}, {cy}, {r})")

#         # 生成最终文本格式
#         output_text = "Line:\n" + "\n".join(lines) + "\n\nCircle:\n" + "\n".join(circles)

#     return output_text
    


