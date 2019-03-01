# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/1 0001 下午 2:24
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用pytorch 实现一个线性回归的例子
"""
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

#torch.linspace(-1,1,100)生成一个100个元素的行向量
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1) # x tensor(100,1)
print(x)
y=x.pow(2)+0.2*torch.rand(x.size()) #y tensor (100,1)

#将数据转化为变量
x,y=Variable(x),Variable(y)

#画图
plt.scatter(x.data.numpy(),y.data.numpy())
#plt.show()

#建立神经网络
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net, self).__init__()
        #定义每层使用什么形式
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        #正向传播输入值
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(n_feature=1,n_hidden=10,n_output=1)
print(net) #打印网络的机构

#训练网络
optimizer=torch.optim.SGD(net.parameters(),lr=0.1)
loss_func=torch.nn.MSELoss() #均方差损失

for i in range(1000):
    prediction=net(x) #传入数据 预测输出值
    loss=loss_func(prediction,y)#计算两者的误差

    optimizer.zero_grad() #清空上一步的残余参数值
    loss.backward() #误差反向传播
    optimizer.step() # 更新参数

    #可视化训练过程
    if i%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.3f'%loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.1)




