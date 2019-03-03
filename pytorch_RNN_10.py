# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/2 0002 下午 3:18
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用Pytorch 框架实现rnn网络
"""
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import  torch.utils.data as DATA
torch.manual_seed(1001)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__() # if use nn.RNN(), it hardly learns
        self.rnn=nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,        # if use nn.RNN(), it hardly learns
            num_layers=1,          # number of rnn layer
            batch_first=True,      # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out=nn.Linear(64,10)

    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out,(h_n,h_c)=self.rnn(x,None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out=self.out(r_out[:,-1,:])
        return out

if __name__ == '__main__':

    #超参数
    EPOCHS=1     # train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE=32
    TIME_STEP=28 #rnn time step / image height
    INPUT_SIZE=28# rnn input size / image width
    LR=0.001      # learning rate
    DOWNLOAD_MNIST=False #set to True if haven't download the data

    #mnist 手写数字集
    train_data=dsets.MNIST(
        root='/data',
        train=True,
        transform=transform.ToTensor(),
        download=DOWNLOAD_MNIST
    )

    #plot one example
    print(train_data.train_data.size())  # (60000, 28, 28)
    print(train_data.train_labels.size()) # (60000)
    plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
    plt.title('%i'%train_data.train_labels[0])
    plt.show()

    #data loader for easy mini-batch return for training
    train_loader=DATA.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

    #Convert test to variable,pick 2000 samples to speed up the testing
    test_data=dsets.MNIST(root='/data',train=False,transform=transform.ToTensor())
    test_x=test_data.test_data.type(torch.FloatTensor)[:2000]/255 # shape (2000, 28, 28) value in range(0,1)
    test_y=test_data.test_labels.numpy()[:2000] # covert to numpy array

    rnn=RNN()
    print('rnn structure:',rnn)

    optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
    loss_func=nn.CrossEntropyLoss()

    #training and testing
    for epoch in range(EPOCHS):
        for step,(batch_x,batch_y) in enumerate(train_loader):
            batch_x=batch_x.view(-1,28,28) # reshape x to (batch, time_step, input_size)
            output=rnn(batch_x)

            loss=loss_func(output,batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%50==0:
                test_output=rnn(test_x)
                pred_y=torch.max(test_output,1)[1].data.numpy()
                accuracy=float((pred_y==test_y).astype(int).sum())/ float(test_y.size)
                print('Epoch:',epoch,"|train loss:%.4f"%loss.data.numpy(),"|test accuracy:%.3f"%accuracy)

    #print 10 predictions from test data
    test_output=rnn(test_x[:10].view(-1,28,28))
    test_y=torch.max(test_output,1)[1].data.numpy()
    print(pred_y,'Prediction number')
    print(test_y[:10],'real number')



