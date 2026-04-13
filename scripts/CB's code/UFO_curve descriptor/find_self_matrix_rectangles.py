"""
找出不等式矩阵 self-matrix 中的最大 = 矩形
Self-matrix: P0 vs P0, P1 vs P1, etc. (对角线上的区域)

输出格式：
- Configuration ID
- Point ID (0, 1, 2, ...)
- Start Timestamp
- End Timestamp  
- Dimension (X or Y)
- Descriptor
"""

import pandas as pd
import numpy as np
import re

def parse_matrix_string(matrix_str):
    """从 CSV 字符串解析 numpy 矩阵"""
    numbers = re.findall(r'(\d+)', matrix_str)
    if numbers:
        n = int(np.sqrt(len(numbers)))
        return np.array([int(x) for x in numbers]).reshape(n, n)
    return None

def find_max_rectangle(matrix, target_value=1):
    """
    使用高度数组栈算法找出矩阵中最大面积的矩形
    返回最大矩形的 (top, left, height, width, area)
    """
    rows, cols = matrix.shape
    heights = np.zeros(cols, dtype=int)
    max_rect = {'area': 0, 'top': 0, 'left': 0, 'height': 0, 'width': 0}
    
    for i in range(rows):
        # 更新高度数组
        for j in range(cols):
            if matrix[i, j] == target_value:
                heights[j] += 1
            else:
                heights[j] = 0
        
        # 计算当前行的最大矩形
        stack = []
        for j in range(cols):
            h_idx = j
            while stack and stack[-1][1] > heights[j]:
                index, height = stack.pop()
                width = j - index
                area = height * width
                top = i - height + 1
                
                if area > max_rect['area']:
                    max_rect = {
                        'top': top, 'left': index, 'height': height,
                        'width': width, 'area': area
                    }
                h_idx = index
            
            if heights[j] > 0:
                stack.append((h_idx, heights[j]))
        
        # 处理栈中剩余元素
        for index, height in stack:
            width = cols - index
            area = height * width
            top = i - height + 1
            if area > max_rect['area']:
                max_rect = {
                    'top': top, 'left': index, 'height': height,
                    'width': width, 'area': area
                }
    
    return max_rect

def get_descriptor(con_id, df_ineq):
    """获取 descriptor (PDPg 的类型)"""
    # TODO: 从配置或文件名推断 descriptor
    # 暂时使用默认值
    return "PDPg_fundamental"

# ============ 主程序 ============

CSV_INEQ_PATH = r"d:\OneDrive - UGent\PhD\PDP\UFO\A2\PDP_results\test_CB\Df_con_tst_xineq_yineq.csv"
OUTPUT_PATH = r"d:\OneDrive - UGent\PhD\PDP\PDP-Analysis\self_matrix_max_rectangles.csv"

# 读取数据集获取 POI 和时间戳信息
df_dataset = pd.read_csv(r"d:\OneDrive - UGent\PhD\PDP\UFO\A2\PDP_results\test_CB\Df_dataset.csv")
n_poi = int(df_dataset['poiID'].max() + 1)

# 从第一个不等式矩阵获取矩阵大小
df_ineq = pd.read_csv(CSV_INEQ_PATH)
first_row = df_ineq.iloc[0]
x_matrix_str = first_row['xineqID']
numbers = re.findall(r'(\d+)', x_matrix_str)
matrix_size = int(np.sqrt(len(numbers)))
window_length_tst = matrix_size // n_poi

print("Configuration: {} POIs, {} timestamps per window".format(n_poi, window_length_tst))
print()

# 初始化输出文件
with open(OUTPUT_PATH, 'w') as f:
    f.write('conID,poiID,start_tst,end_tst,dimension,area,height,width\n')

results = []

# 处理每一行
for idx, row in df_ineq.iterrows():
    con_id = int(row['conID'])
    tst_id = int(row['tstID'])
    descriptor = get_descriptor(con_id, df_ineq)
    
    print("Processing Config {}, Timestamp {} ...".format(con_id, tst_id))
    
    # 处理 x 维度
    x_matrix_str = row['xineqID']
    x_matrix = parse_matrix_string(x_matrix_str)
    
    # 处理 y 维度
    y_matrix_str = row['yineqID']
    y_matrix = parse_matrix_string(y_matrix_str)
    
    if x_matrix is None or y_matrix is None:
        print("  Skip: matrix parse failed")
        continue
    
    # 对每个 POI 的 self-matrix 找最大矩形
    for poi_id in range(n_poi):
        # 计算 self-matrix 在完整矩阵中的位置
        start_idx = poi_id * window_length_tst
        end_idx = (poi_id + 1) * window_length_tst
        
        # 提取 self-matrix
        x_self = x_matrix[start_idx:end_idx, start_idx:end_idx]
        y_self = y_matrix[start_idx:end_idx, start_idx:end_idx]
        
        # 查找 X 维度的最大 = 矩形
        rect_x = find_max_rectangle(x_self, target_value=1)
        
        if rect_x['area'] > 0:
            # 在 self-matrix 中的位置就对应时间戳
            # self-matrix 的每一行对应一个时间戳
            tst_start = tst_id + rect_x['top']
            tst_end = tst_id + rect_x['top'] + rect_x['height'] - 1
            
            results.append({
                'conID': con_id,
                'poiID': poi_id,
                'start_tst': tst_start,
                'end_tst': tst_end,
                'dimension': 'X',
                'area': rect_x['area'],
                'height': rect_x['height'],
                'width': rect_x['width']
            })
            
            if rect_x['area'] > 1:
                print("  X dim - POI {}: {}x{} area={} ({}-{})".format(
                    poi_id, rect_x['height'], rect_x['width'], rect_x['area'], 
                    tst_start, tst_end))
        
        # 查找 Y 维度的最大 = 矩形
        rect_y = find_max_rectangle(y_self, target_value=1)
        
        if rect_y['area'] > 0:
            tst_start = tst_id + rect_y['top']
            tst_end = tst_id + rect_y['top'] + rect_y['height'] - 1
            
            results.append({
                'conID': con_id,
                'poiID': poi_id,
                'start_tst': tst_start,
                'end_tst': tst_end,
                'dimension': 'Y',
                'area': rect_y['area'],
                'height': rect_y['height'],
                'width': rect_y['width']
            })
            
            if rect_y['area'] > 1:
                print("  Y dim - POI {}: {}x{} area={} ({}-{})".format(
                    poi_id, rect_y['height'], rect_y['width'], rect_y['area'],
                    tst_start, tst_end))

# 保存结果
output_df = pd.DataFrame(results)

# 过滤掉面积为 1 的矩形（只保留边长 >= 2 的）
output_df = output_df[output_df['area'] > 1]

output_df = output_df.sort_values(['conID', 'poiID', 'area'], ascending=[True, True, False])

# 只保留每个 (conID, poiID, dimension) 组合的最大矩形
unique_results = output_df.drop_duplicates(subset=['conID', 'poiID', 'dimension'], keep='first')

unique_results.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to: {}".format(OUTPUT_PATH))
print("Found {} self-matrix max rectangles (area > 1)".format(len(unique_results)))
print("\nSummary (sorted by area):")
summary = unique_results[['conID', 'poiID', 'dimension', 'start_tst', 'end_tst', 'area']].sort_values('area', ascending=False)
print(summary.to_string())
