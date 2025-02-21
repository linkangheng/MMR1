import os
import pandas as pd
from PIL import Image
import io
import numpy as np

# 设置 Parquet 文件所在文件夹路径
DATASET_DIR = "/data/dataspace/slow_percept-sub1-ori_form-rl"


# 获取所有 Parquet 文件
parquet_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(".parquet")]

# 检查 image 是否能正确解析
def check_image_validity(image_dict):
    try:
        if isinstance(image_dict, dict) and "bytes" in image_dict:
            Image.open(io.BytesIO(image_dict["bytes"]))  # 解析 PNG
            return True
    except:
        return False
    return False

# 检查 solution 格式
def check_solution_format(sol):
    if isinstance(sol, str) and "Line" in sol:
        return True
    return False

# # 检查 solution 数值
# def check_numeric_values(sol):
#     try:
#         if isinstance(sol, dict) and "Line" in sol:
#             for line in sol["Line"]:
#                 if not all(isinstance(coord, (int, float)) and np.isfinite(coord) for coord in line):
#                     return False
#             return True
#     except:
#         return False
#     return False

# 逐个文件检查
for parquet_file in parquet_files:
    print(f"Checking file: {parquet_file}")
    
    try:
        df = pd.read_parquet(parquet_file)

        # 1. 检查 image 列
        invalid_images = df[~df["image"].apply(check_image_validity)]
        if not invalid_images.empty:
            print(f"❌ {parquet_file}: {len(invalid_images)} damaged images")

        # 2. 检查 solution 是否为空
        missing_solutions = df[df["solution"].isnull()]
        if not missing_solutions.empty:
            print(f"⚠️ {parquet_file}: {len(missing_solutions)} missing solutions")

        # 3. 检查 solution 是否为字典格式
        invalid_solutions = df[~df["solution"].apply(check_solution_format)]
        if not invalid_solutions.empty:
            print(f"❌ {parquet_file}: {len(invalid_solutions)} incorrect solution format")

        # # 4. 检查 solution 数值是否合法
        # invalid_numeric = df[~df["solution"].apply(check_numeric_values)]
        # if not invalid_numeric.empty:
        #     print(f"❌ {parquet_file}: {len(invalid_numeric)} solutions with non-numeric values")

    except Exception as e:
        print(f"❌ Failed to process {parquet_file}: {e}")

print("✅ Parquet data check complete!")