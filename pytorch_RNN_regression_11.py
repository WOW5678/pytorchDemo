# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/3 0003 上午 9:37
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: RNN 实现回归预测
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32, # rnn hidden unit
            num_layers=1,   # number of rnn layer
            batch_first=True # input & output will has batch size as 1s dimension.
                             #  e.g. (batch, time_step, input_size)
        )
        self.out=nn.Linear(32,1)

    def forward(self,x,h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out,h_state=self.rnn(x,h_state)

        outs=[]   # save all predictions
        for time_step in range(r_out.size(1)): # calculate output for each time step
            outs.append(self.out(r_out[:,time_step,:]))
        x= torch.stack(outs, dim=1)
        print(x.size()) #([1, 10, 1])
        return torch.stack(outs,dim=1),h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state

        # or even simpler, since nn.Linear can accept inputs of any dimension
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs


torch.manual_seed(1001)

if __name__ == '__main__':
    #超参数
    TIME_step=10
    INPUT_SIZE=1
    LR=0.02

    #show data
    steps=np.linspace(0,np.pi*2,dtype=np.float32)
    x_np=np.sin(steps)
    y_np=np.cos(steps)
    plt.plot(steps,y_np,'r-',label='target(cos)')
    plt.plot(steps,x_np,'b-',label='input(sin)')
    plt.legend()
    plt.show()

    rnn=RNN()
    print(rnn)
    optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
    loss_func=nn.MSELoss()

    h_state=None  # for initial hidden state
    plt.figure(1,figsize=(12,5))
    plt.ion()

    for step in range(100):
        start,end=step*np.pi,(step+1)*np.pi
        #use son predicts cos
        # float32 for converting torch FloatTensor
        steps=np.linspace(start,end,TIME_step,dtype=np.float32,endpoint=False)
        x_np=np.sin(steps)
        y_np=np.cos(steps)

        x=torch.from_numpy(x_np[np.newaxis,:,np.newaxis])
        y=torch.from_numpy(y_np[np.newaxis,:,np.newaxis])

        prediction,h_state=rnn(x,h_state)
        print('prediction size:',prediction.size()) #[1, 10, 1]
        h_state=h_state.data  # repack the hidden state, break the connection from last iteration

        loss=loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #plotting
        plt.plot(steps,y_np.flatten(),'r-')
        plt.plot(steps,prediction.data.numpy().flatten(),'b-')
        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()


