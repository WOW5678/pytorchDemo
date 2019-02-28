# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/28 0028 下午 6:34
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: pytorch 与numpy之间的转换
"""
import torch
import numpy as np

np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)
tensor2numpy=torch_data.numpy()
print('numpy data:',np_data,
      '\n torch tensor:',torch_data,
      '\n tensor to numpy:',tensor2numpy)
#torch 中的数学运算
data=[-1,-2,-2,2]
tensor=torch.FloatTensor(data) #转化为32位浮点数
print(np.abs(data))
print(torch.abs(tensor)) #tensor([1., 2., 2., 2.])
#三角函数
print(np.sin(data))
print(torch.sin(tensor))

#mean均值
print(np.mean(data))
print(torch.mean(tensor))

#矩阵的乘法
data=[[1,2],[3,4]]
tensor=torch.FloatTensor(data)
print(np.matmul(data,data))
#tensor([[ 7., 10.],
#        [15., 22.]])
print(torch.mm(tensor,tensor))

#!!!!以下是错误的方法！！！
data=np.array(data)
# [[ 7 10]
#  [15 22]] 在numpy 中是可行的 dot 就是矩阵乘 但是torch 中是不可行的
print(data.dot(data))
#新版本中(>=0.3.0), 关于 tensor.dot() 只能针对于一维的数组.
print(tensor.dot(tensor))


