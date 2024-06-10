# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:58:25 2023

@author: kernel
"""
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('E:/F-kac-半线性/logs/test_training_history.csv')

# 过滤数据，只保留step为0-2000的最后一行
mask = (df['step'] >= 0) & (df['step'] <= 10000)
filtered_data = df[mask]

# 找到每个 'step' 值的最后一行
final_data = filtered_data.groupby('step').last()

# 删除重复的行
final_data.drop_duplicates(inplace=True)

# 输出结果
print(final_data)

ralative_error = np.abs((final_data['target_value'] - 21.299 )/21.299)