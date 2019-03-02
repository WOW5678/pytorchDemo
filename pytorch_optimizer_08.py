# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/2 0002 上午 9:25
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用pytorch框架尝试各种优化器 为数据和模型寻找最优的优化器
"""
import torch
import torch.utils.data as DATA
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(1001) #为了结果的重现


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == '__main__':

    LR=0.001
    BATCH_SIZE=32
    EPOCH=2

    #生成一些伪造的数据
    x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
    y=x.pow(2)+0.1*torch.normal(torch.zeros(*x.size()))
    print(y.size()) #torch.Size([1000, 1])

    #画出数据集
    plt.scatter(x.numpy(),y.numpy())
    plt.show()

    #使用上节内容提到的data loader
    #torch_data=DATA.TensorDataset(x,y)
    torch_data=DATA.TensorDataset(x, y)
    loader=DATA.DataLoader(
        dataset=torch_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
    )

    #每个优化器优化一个神经网络



    #为每个优化器创建一个net
    net_SGD=Net()
    net_Momentum=Net()
    net_RMSprop=Net()
    net_Adam=Net()
    nets=[net_SGD,net_Momentum,net_RMSprop,net_Adam]

    opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
    opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
    opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
    opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
    optimizer=[opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

    loss_func=torch.nn.MSELoss()
    lossese_his=[[],[],[],[]] #记录训练过程中不同神经网络的loss

    #训练 并展示结果
    for epoch in range(EPOCH):
        print('epoch:',epoch)
        for step,(b_x,b_y) in enumerate(loader):
            #对每个优化器 优化属于他的神经网络
            for net,opt,l_his in zip(nets,optimizer,lossese_his):
                output=net(b_x) #得到每个网络的输出
                loss=loss_func(output,b_y)

                #三部曲 必不可少 而且一般不变化
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())
    #画图
    for i in  range(4):
        plt.plot(range(len(lossese_his[i])),lossese_his[i],label=optimizer[i])
    plt.legend()
    plt.show()
