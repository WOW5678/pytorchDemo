# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/28 0028 下午 6:58
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: torch中的变量的使用

"""
import  torch
from torch.autograd import Variable


tensor=torch.FloatTensor([[1,2],[3,4]])
#requires_grad是参不参与误差反向传播 要不要计算梯度
varibale=Variable(tensor,requires_grad=True)
# tensor([[1., 2.],
#         [3., 4.]])
print(tensor)
# tensor([[1., 2.],
#         [3., 4.]], requires_grad=True)
print(varibale)
#对比一下tensor的计算和variable的计算
t_out=torch.mean(tensor*tensor)
v_out=torch.mean(varibale*varibale)
#tensor(7.5000) tensor(7.5000, grad_fn=<MeanBackward1>)
print(t_out,v_out)

v_out.backward() #模拟v_out的误差反向传递
print(varibale.grad) # variable的梯度

#获取variable里面的向量
# 直接print(variable)只会输出 Variable 形式的数据, 在很多时候是用不了的(比如想要用 plt 画图), 所以我们要转换一下, 将它变成 tensor 形式.
# 利用Variable的data属性得到tensor对象，然后转换成numpy数据

print(varibale)
print(varibale.data) #tensor形式
print(varibale.data.numpy()) #numpy形式