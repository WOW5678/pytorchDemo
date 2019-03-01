# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/1 0001 下午 2:54
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用pytorch训练处的模型的保存和提取
"""
import torch
from torch.autograd import Variable
torch.manual_seed(1) #为了重现模型的结果

#假数据
x=torch.unsqueeze(torch.linspace(-1,1,100),1)
y=x.pow(2)+0.2*torch.rand(x.size())
#将x,y都变成变量
x=Variable(x)
y=Variable(y)

def save():
    #建网络
    net1=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer=torch.optim.SGD(net1.parameters(),lr=0.1)
    loss_func=torch.nn.MSELoss()

    #训练
    for t in range(100):
        prediction=net1(x)
        loss=loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #保存网络 两种途径
    torch.save(net1,'net.pkl')
    torch.save(net1.state_dict(),'net_param.pkl') #只保存网络中的参数

#提取网络 该方法可以提取整个网络
def restore_net():
    #restore整个网络net1 to net2
    net2=torch.load('net.pkl')
    prediction=net2(x)
    print(prediction)
##只提取网络参数

def restore_params():
    #新建net3网络 需要与网络1 完全一样
    net3=torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    #将保存的参数赋值到net3中
    net3.load_state_dict(torch.load('net_param.pkl'))
    prediction=net3(x)
    print(prediction)

#显示结果
save()
#提取真个网络
restore_net()
#提取网络参数
restore_params()

