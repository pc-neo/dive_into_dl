import torch
import os
import pandas as pd
import numpy as np

file_path = os.path.join("test.csv")
# with open(file_path, "w") as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')

data = pd.read_csv(file_path, dtype={
    "NumRooms": float,
    "Alley": str,
    "Price": float
})
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2] 
nan_cnts = inputs.isnull().sum().sort_values(ascending=False)
most_nan_columns = nan_cnts.index[0]
print(most_nan_columns)
inputs = inputs.drop(columns=[most_nan_columns])
numeric_columns = inputs.select_dtypes(include=['float64', 'int64']).columns
inputs[numeric_columns] = inputs[numeric_columns].fillna(inputs[numeric_columns].mean())
print(inputs)    

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)


X = torch.tensor(inputs.to_numpy(dtype=np.float32))
y = torch.tensor(outputs.to_numpy(dtype=np.float32))
print(X)
print(y)