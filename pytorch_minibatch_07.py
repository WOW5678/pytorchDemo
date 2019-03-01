# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/1 0001 下午 6:21
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 用pytorch 框架实现批训练数据
"""
import torch
import torch.utils.data as DATA
torch.manual_seed(1001) #复现结果


if __name__ == '__main__':

    BATCH_SIZE=5 #批训练的数据个数

    x=torch.linspace(1,10,10)
    y=torch.linspace(10,1,10)

    #先转化成torch能够识别的Dataset
    torch_dataset= DATA.TensorDataset(x,y)
    #torch_dataset_y=DATA.TensorDataset(y)
    #把dataset 放入DataLoader中
    loader=DATA.DataLoader(
        dataset=torch_dataset, #torch TensorDataset形式
        batch_size=BATCH_SIZE, #minibatch size
        shuffle=True, #是否打乱数据
        num_workers=2, #多线程读取数据
    )

    for epoch in range(3):
        for step ,(batch_x,batch_y) in enumerate(loader):
            #假设这里就是你训练的地方

            #打印数据
            print('Epoch:',epoch,"|step:",step,"|batch x:",
                  batch_x.numpy(),"|batch_y:",batch_y.numpy())

