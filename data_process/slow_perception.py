import os
import json
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

# 目录列表
root_dirs = ["dataset/jihe_train_data/sub1"] # "dataset/jihe_train_data/sub2"
output_dir = "/data/dataspace/slow_percept-sub1-rl"  # 存储 Parquet 的目录
output_dir2 = "/data/dataspace/slow_percept-sub1-ori_form-rl"  # 存储 Parquet 的目录
batch_size = 2600  # 每 2600 条数据存储一个 parquet 文件


# **坐标转换**
def coor2txt(coordinates):
    lines = []
    circles = []

    for item in coordinates:
        if "Line" in item:
            for line in item["Line"]:
                x1, y1, x2, y2 = line
                lines.append(f"({x1}, {y1}) -- ({x2}, {y2})")

        if "Circle" in item:
            for circle in item["Circle"]:
                cx, cy, r = circle
                circles.append(f"({cx}, {cy}, {r})")

    return "Line:\n" + "\n".join(lines) + "\n\nCircle:\n" + "\n".join(circles)

def save_parquet(output_dir, file_name, data_list):
    output_path = os.path.join(output_dir, file_name)

    # **转换 DataFrame 并写入 Parquet**
    df = pd.DataFrame(data_list)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)

    print(f"✅ Saved {output_path}")


# **统计所有 JSON 文件数量**
total_files = 0
json_files = []

for rootdir in root_dirs:
    json_dir = f"{rootdir}/jsons"
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_files.append((rootdir, filename))
            total_files += 1

# 计算总的 Parquet 文件数
total_shards = (total_files // batch_size) + (1 if total_files % batch_size != 0 else 0)

# **开始处理数据**
data_list = []
data_list2 = []
file_count = 0

for rootdir, filename in tqdm(json_files):
    json_path = os.path.join(rootdir, "jsons", filename)
    img_path = os.path.join(rootdir, "imgs", filename.replace(".json", ".png"))

    # 读取图片
    image_bytes = None
    if os.path.exists(img_path):
        with open(img_path, "rb") as img_file:
            image_bytes = img_file.read()

    # 读取 JSON 并转换 solution
    with open(json_path, "r") as f:
        json_data = json.load(f)

    

    solution_text = coor2txt(json_data)
    solution = f"<answer>\n{solution_text}\n</answer>"
    solution2 = f"<answer>\n{str(json_data)}\n</answer>"

    # **构造 problem**
    problem = "Describe the start and end points of each line and the position (center, radius) of the circle in the image."

    # import ipdb;ipdb.set_trace()
    # **存储数据**
    data_list.append({"image": {'bytes': image_bytes}, "problem": problem, "solution": solution})
    data_list2.append({"image": {'bytes': image_bytes}, "problem": problem, "solution": solution2})

    # **每 batch_size (2600) 存一次**
    if len(data_list) >= batch_size:
        file_name = f"train-{file_count:05d}-of-{total_shards:05d}.parquet"
        save_parquet(output_dir, file_name, data_list)
        save_parquet(output_dir2, file_name, data_list2)

        # **重置 data_list**
        data_list = []
        file_count += 1

# **存储剩余数据**
if data_list:
    file_name = f"train-{file_count:05d}-of-{total_shards:05d}.parquet"
    save_parquet(output_dir, file_name, data_list)
    save_parquet(output_dir2, file_name, data_list2)