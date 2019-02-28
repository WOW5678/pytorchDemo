"""
 @Time    : 2019/2/28 0028 下午 7:10
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用pytorch完成一个简单的神经网络分类
"""
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

#假数据
n_data=torch.ones(100,2) #数据的基本形态
x0=torch.normal(2*n_data,1)
# print(n_data)#tensor shape=(100,2)
# print(x0) #tensor shape=(100,2)
y0=torch.zeros(100) #tensor shape=(100,1)

x1=torch.normal(-2*n_data,1)
y1=torch.ones(100)

#torch.cat是在合并数据时使用的
x=torch.cat((x0,x1),0).type(torch.FloatTensor)#转换成32位浮点型tensor
y=torch.cat((y0,y1),0).type(torch.LongTensor) #LongTensor=64bit integer
print(x.shape,y.shape)
#torch只能在variable上训练，因此把他们变成variable
x,y=Variable(x),Variable(y)
#画图
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

#建立神经网络
import torch.nn.functional as F #激励函数都在这里

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__() #继承 __init__的功能
        self.hidden=torch.nn.Linear(n_feature,n_hidden) #隐藏层线性输出
        self.out=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        #正向传播输入的值 神经网络分析输出值
        x=F.relu(self.hidden(x)) #激励函数（隐藏层的线性值）
        x=self.out(x) #输出值，但是这个不是预测值，预测值需要另外计算
        return x

net=Net(n_feature=2,n_hidden=10,n_output=2) #几个类别 就是几个output
print(net) #输出net的结构

#训练网络
optimizer=torch.optim.SGD(net.parameters(),lr=0.002) #出入net的所有参数
#算误差的时候 注意真实值不是one-hot形式的，而是1Dtensor(batch,)
#但是预测值是2D tensor (batch,n_classes)
loss_fn=torch.nn.CrossEntropyLoss()

#画图
plt.ion()
plt.show()
for t in range(100):
    out=net(x) #喂给Net训练数据 输出分析值
    print(out.shape,y.shape) #[200,2] [200]
    loss=loss_fn(out,y) #计算两者的误差

    optimizer.zero_grad() #清空上一步的残余跟新参数
    loss.backward() #误差反向传播 计算参数更新值
    optimizer.step() #将参数更新值 施加到net的parameters上

    #可视化训练过程
    if t%2==0:
        plt.cla()
        #过了一道softmax的激励函数之后的最大概率才是预测值
        prediction=torch.max(F.softmax(out),1)[1]
        pred_y=prediction.data.numpy().squeeze()
        target_y=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='RdYlGn')
        accuracy=sum(pred_y==target_y)/200 #预测值中有多少和真实值一样
        plt.text(1.5,-4,'Accuracy=%.2f'%accuracy,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ion()
plt.show()
