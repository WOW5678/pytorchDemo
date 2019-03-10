# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/10 0010 上午 9:07
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用pytorch框架实现一个基础版的GAN
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1001)
np.random.seed(1001)


def artist_works():
    a=np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
    print(a.shape) #(64,1)
    paintings=a*np.power(PAINT_POINTS,2)+(a-1)
    paintings=torch.from_numpy(paintings).float()
    return paintings #(64,1)

if __name__ == '__main__':
    #超参数
    BATCH_SIZE=64
    LR_G=0.0001
    LR_D=0.0001
    N_IDEAS=5
    ART_COMPONENTS=15
    PAINT_POINTS=np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)])
    print(PAINT_POINTS.shape)  #(64, 15)

    #定义G网络
    G=nn.Sequential(
        nn.Linear(N_IDEAS,128), #random ideas (could from normal distribution)
        nn.ReLU(),
        nn.Linear(128,ART_COMPONENTS), # making a painting from these random ideas
    )

    #定义D网络
    D=nn.Sequential(
        nn.Linear(ART_COMPONENTS,128), # making a painting from these random ideas
        nn.ReLU(),
        nn.Linear(128,1),
        nn.Sigmoid(),# tell the probability that the art work is made by artist
    )

    opt_D=torch.optim.Adam(D.parameters(),lr=LR_D)
    opt_G=torch.optim.Adam(G.parameters(),lr=LR_G)

    #画动态图
    plt.ion() #something about continuous plotting

    for step in range(10000):
        artst_paintings=artist_works()
        G_ideas=torch.randn(BATCH_SIZE,N_IDEAS)
        G_patings=G(G_ideas)

        prob_artist0=D(artst_paintings)#D try to increase this prob
        prob_artist1=D(G_patings)       # D try to reduce this prob

        D_loss=-torch.mean(torch.log(prob_artist0)+torch.log(1-prob_artist1))
        G_loss=torch.mean(torch.log(1-prob_artist1))

        #开始优化
        opt_D.zero_grad()
        #保留变量参数的原因是因为误差反向传播计算到prob_artist0，
        # prob_artist1的时候都会用到D网络的参数
        D_loss.backward(retain_graph=True)
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if step%50==0:
            plt.cla()
            plt.plot(PAINT_POINTS[0],G_patings.data.numpy()[0],c='#4AD631', lw=3, label='Generated painting',)
            plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                     fontdict={'size': 13})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=10)
            plt.draw()
            plt.pause(0.01)
    plt.ioff()
    plt.show()

