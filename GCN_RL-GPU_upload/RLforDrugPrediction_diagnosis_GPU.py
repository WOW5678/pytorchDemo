# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/20 0020 下午 7:35
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:利用RL结合GCN 进行多种药物的预测
 使用诊断数据对病人进行建模
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import collections
import random
from sklearn.model_selection import train_test_split
from RL_data_util import *
from GCN import *

torch.manual_seed(1001)
random.seed(1001)
np.random.seed(1001)

device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('device:',device)

class CNN(nn.Module):
    def __init__(self,vocab_size,emb_size,num_channels,hidden_dim,dropout):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(embedding_dim=emb_size,num_embeddings=vocab_size)
        self.conv=nn.Sequential(
            nn.Conv1d(           #input shape (1, 28, 28)
                in_channels=emb_size,   #input height
                out_channels=num_channels, # n_filters
                kernel_size=3,   # filter size
                stride=2,        # filter movement/step
                                 # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),              # output shape (16, 28, 28)
            nn.ELU(inplace=True),   # activation
            #nn.MaxPool2d(kernel_size=2),# choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Conv1d(  # input shape (1, 28, 28)
                in_channels=num_channels,  # input height
                out_channels=num_channels,  # n_filters
                kernel_size=3,  # filter size
                stride=2,  # filter movement/step
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ELU(inplace=True),  # activation
            #nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.dropout=dropout
        self.out=nn.Linear(num_channels,hidden_dim)

    def forward(self,x):
        #print('x:',type(x))
        x_emb=self.embedding(x).unsqueeze(0).permute(0,2,1)
        #print('x_emb:',x_emb.shape) #[1, 100, 30]
        x = self.conv(x_emb)
        # average and remove the extra dimension
        remaining_size = x.size(dim=2)
        features = F.avg_pool1d(x, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self.dropout)
        output = self.out(features)
        return output

class DQN(nn.Module):
    def __init__(self,state_size,hidden1,hidden2,action_size):
        super(DQN, self).__init__()
        # 一个简单的三层的感知器网络用来根据状态做决策
        self.fc1=nn.Linear(state_size,hidden1)
        self.fc2=nn.Linear(hidden1,hidden2)
        self.fc3=nn.Linear(hidden2,action_size)
        self.learn_step_counter=0 #用于判断何时更新target网络
    def forward(self,x):
        #print(x.shape)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        output=self.fc3(x)
        #output=F.sigmoid(output)
        return output

class Agent(object):
    def __init__(self,state_size,action_size,layer_sizes):
        super(Agent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=1000)
        self.gamma = 1.0  # 计算未来奖励的折扣率
        self.epsilon = 0.9  # agent最初探索环境选择action的探索率
        self.epsilon_min = 0.01  # agent控制随机探索的阈值
        self.epsilon_decay = 0.995  # 随着agent做选择越来越好，降低探索率
        self.learning_rate = 0.01
        self.cnn = CNN(patient_vocab_size+2,EMB_SIZE,128,EMB_SIZE,0.5).cuda(1)
        self.rgcn= RGCN(layer_sizes,drug_vocab_size).cuda(1)
        self.model=DQN(state_size,512,512,action_size).cuda(1)
        self.target_model =DQN(state_size,512,512,action_size).cuda(1)
        self.model_params = list(self.cnn.parameters()) +list(self.rgcn.parameters())+ list(self.model.parameters())
        self.optimizier=torch.optim.Adam(self.model.parameters(),lr=LR)
        self.loss=nn.MSELoss()
        self.update_target_model()

    def reset(self,x):
        #得到每个电子病历数据的表示
        f = self.cnn(x)
        #print('g:',f.shape)
        return f

    def act(self,state):
        #根据state 选择action
        if np.random.rand()<self.epsilon:
            while True:
                action=random.randrange(self.action_size)
                if action not in selectedAction:
                    return action
        output=self.model(state)
        while True:
            action = torch.max(output, 1)[1]
            if action not in selectedAction:
                return action
            else:
                output[0][action]=-999999

    def step(self,action,selectedAction):
        #根据该action进行奖励
        if int(action) in y :
            reward=2
        else:
            reward=-1
        #更新新的药物图谱
        adjacencies=getADJ(action,selectedAction,drug2id)
        adjacencies = list(map(get_torch_sparse_matrix, adjacencies, [device] * len(adjacencies)))
        _, g = self.rgcn(adjacencies)

        THC_CACHING_ALLOCATOR = 0
        return reward,g

    def remember(self,state,action,reward,next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self,BATCH_SIZE):
        print('learning_step_counter:', self.model.learn_step_counter)
        # 没训练一次 都将模型learn_step_counter加一 并且判断是否需要更新target网络
        if self.model.learn_step_counter%TARGET_UPDATE_ITER==0:
            print('Update target model.')
            self.update_target_model()
        self.model.learn_step_counter+=1
        batch_idx=np.random.choice(len(self.memory),BATCH_SIZE)
        #print(batch_idx)
        b_s=[]
        b_action = []
        b_reward=[]
        b_next_s=[]
        for id in batch_idx:
            b_s.append(self.memory[id][0])
            b_action.append(self.memory[id][1])
            b_reward.append(self.memory[id][2])
            b_next_s.append(self.memory[id][3])
        #将这些数据转变成tensor对象
        b_s=torch.cat(b_s,0).cuda(1)
        b_reward=torch.FloatTensor(b_reward).cuda(1)
        b_action=torch.LongTensor(b_action).cuda(1)
        b_next_s=torch.cat(b_next_s,0).cuda(1)

        b_action=b_action.unsqueeze(1)
        q_eval=self.model(b_s).gather(1,b_action)
        q_next=self.target_model(b_next_s)
        q_target=b_reward+GAMMA*q_next.max(1)[0].view(BATCH_SIZE,1)
        loss=self.loss(q_eval,q_target)

        self.optimizier.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizier.step()

        #保存下训练过程中的损失函数值
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

        del batch_idx,b_s,b_action,b_reward,b_next_s
        del q_eval,q_target,q_next
        return loss

    def update_target_model(self):
        #加载self.model的参数到target_model中
        self.target_model.load_state_dict(self.model.state_dict())

class Evaluate(object):
    def __init__(self,X_,Y_):
        self.X=torch.LongTensor(X_)
        self.Y=Y_

    def evaluate(self):
        # 评估在数据集上的模型表现
        # 其他统计指标
        Jaccard_list = []
        Recall_list = []
        Reward_list = []
        D_D_Num=0
        adjacencies = init_ADJ(drug2id)
        adjacencies = list(map(get_torch_sparse_matrix, adjacencies, [device] * len(adjacencies)))


        for sample, label in zip(self.X, self.Y):
            sampleReward = 0
            y = set(label)  # 因为y中有重复的药物 因此这里转变为集合 去掉重复的药物
            # 得到初始的状态
            sample=sample.cuda(1)
            # 使用GCN对adjaccies进行处理 得到图表示
            _, g = agent.rgcn(adjacencies)
            del adjacencies

            f= agent.reset(sample)
            selectedAction=[]
            state = torch.cat((f, g), 1)
            for step in range(len(y)):
                # 模型根据state 选择action 不在随机探索
                output = agent.model(state)
                while True:
                    action = torch.max(output, 1)[1]
                    if action not in selectedAction:
                        break
                    else:
                        output[0][action] = -999999
                # 执行该action 得到reward 并更新状态
                reward, g = agent.step(action, selectedAction)
                #print('epoch:%d, action:%d, reward:%d' % (e, action, reward))
                next_state = torch.cat((f, g), 1)
                # 将选择的action加入到selectedAction中
                selectedAction.append(int(action))
                sampleReward += reward
                # 将经验放入经验池
                # 记忆这次transition
                agent.remember(state, action, reward, next_state)
            # print('y:', y)
            # print('set(select  edAction):', set(selectedAction))
            jaccard, recall, precision, f_measure = self.evaluate_sample(selectedAction, y)
            Jaccard_list.append(jaccard)
            Recall_list.append(recall)
            Reward_list.append(sampleReward)
            # 判断生成的药物中是否有DDI药物对
            d_d,d_d_num = self.evaluate_ddi(y_pred=selectedAction)
            print('d_d:',d_d)
            if d_d_num>0:
                D_D_Num+=1
        avg_jaccard = sum(Jaccard_list) * 1.0 / len(Jaccard_list)
        avg_recall = sum(Recall_list) * 1.0 / len(Recall_list)
        avg_reward = sum(Reward_list) * 1.0 / len(Reward_list)
        print('epoch:{},avg_jaccard:{},avg_recall:{},avg_reward:{},D_D_Num'.format(e, avg_jaccard, avg_recall, avg_reward,D_D_Num))
        del Jaccard_list,Recall_list,Reward_list
        return avg_reward,avg_jaccard,avg_recall,D_D_Num

    def evaluate_sample(self,y_pred,y_true): #针对单个样本的三个指标的评估结果
        y_pred=set(y_pred)
        jiao_1 = [item for item in y_pred if item in y_true]
        bing_1 = [item for item in y_pred] + [item for item in y_true]
        bing_1 = list(set(bing_1))
        # print('jiao:',jiao_1)
        # print('bing:',bing_1)
        recall = len(jiao_1) * 1.0 / len(y_true)
        precision = len(jiao_1) * 1.0 / len(y_pred)
        jaccard = len(jiao_1) * 1.0 / len(bing_1)

        if recall + precision == 0:
            f_measure = 0
        else:
            f_measure = 2 * recall * precision * 1.0 / (recall + precision)
        print('jaccard:%.3f,recall:%.3f,precision:%.3f,f_measure:%.3f' % (jaccard, recall, precision, f_measure))
        del jiao_1,bing_1
        return jaccard,recall,precision,f_measure

    def evaluate_ddi(self,y_pred):
        y_pred=set(y_pred)
        #根据药物id找到对应的药物名称
        pred_drugs=[drug2id.get(id) for id in y_pred]
        #判断这些药物中是否存在着对抗的药物对

        #对生成的药物进行两两组合
        D_D=[]
        for i in range(len(pred_drugs) - 1):
            for j in range(i + 1, len(pred_drugs)):
                key1 = [pred_drugs[i],pred_drugs[j]]
                key2 = [pred_drugs[j],pred_drugs[i]]

                if key1 in ddiPairs or key2 in ddiPairs:
                    # 记录下来该DDI数据  以便论文中的case Study部分分析
                    D_D.append(key1)
        del pred_drugs,key1,key2
        return D_D,len(D_D)

    def plot_result(self,total_reward,total_recall,total_jaccard):
        # 画图
        import matplotlib.pyplot as plt
        import matplotlib
        # 开始画图
        plt.figure()
        ax = plt.gca()

        epochs = np.arange(len(total_reward))

        plt.subplot(1, 2, 1)
        # 设置横坐标的显示刻度为50的倍数
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.plot(epochs, total_reward, label='Reward')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, total_recall, label='Recall', color='red')
        plt.plot(epochs, total_jaccard, label='Jaccard')
        # 设置横坐标的显示刻度为50的倍数
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.show()  # 展示

if __name__ == '__main__':
    # 1. 加载数据集
    patients, drugList,drugSet = load_data_diagnosis('/home/wangshanshan/GCN_RL-GPU/data/patientInformation_0103_onetime_current.csv')
    #print('drugs:', drugs)
    # 2. 进行单词分割 id化
    X, maxLength, patient_tokenizer= tokenization(patients)
    Y,drug2id=drug_tokenizer(drugList,drugSet)
    #print('Y:',Y[:10])
    patient_vocab_size = len( patient_tokenizer.word_index)
    drug_vocab_size = len(drug2id)

    print('patient_vocab_size:', patient_vocab_size)
    print('drug_vocab_size:', drug_vocab_size)

    # 准备好DDI药物对
    with open('/home/wangshanshan/GCN_RL-GPU/data/newDDIData_filter.csv','r',encoding='utf-8') as f:
        reader=csv.reader(f)
        ddiPairs=[[row[0],row[1]] for row in reader]

    # 3. 分割训练集和验证集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.101, random_state=1)
    # 对训练集再次进行划分 得到训练集和验证集
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1124, random_state=1)
    print('train,val,test:', len(X_train), len(X_val), len(X_test))
    X_train=torch.LongTensor(X_train)

    #删除不再使用的变量
    del X,Y,patients,drugSet,drugList,patient_tokenizer

    # 4. 强化学习部分
    # 超参数参数
    LR=0.001
    EMB_SIZE=128
    Layer_sizes=[50,50]
    state_size = EMB_SIZE+sum(Layer_sizes)
    action_size = drug_vocab_size
    #print('action_size:', action_size)
    agent = Agent(state_size, action_size,layer_sizes=Layer_sizes)
    BATCH_SIZE = 32
    EPISODES = 1000   # 让agent玩游戏的次数
    TARGET_UPDATE_ITER=50
    GAMMA=1

    evl_val = Evaluate(X_val, Y_val)
    #evl_test=Evaluate(X_test,Y_test)

    #写结果文件
    f=open('/home/wangshanshan/GCN_RL-GPU/data/result.csv','a+',newline='',encoding='utf-8')
    writer=csv.writer(f)


    #因为每个样本的初始时有一样的图谱，所以放在循环外面，可以节省计算资源和时间
    # 初始化图
    adjacencies= init_ADJ(drug2id)
    adjacencies = list(map(get_torch_sparse_matrix, adjacencies, [device] * len(adjacencies)))

    for e in range(EPISODES):
        print('epoch:%d'%e)
        epochLoss=[]
        #针对每个EHR
        #使用CNN得到EHR的表示f
        for x,y in zip(X_train,Y_train):
            # 使用GCN对adjaccies进行处理 得到图表示
            _, g = agent.rgcn(adjacencies)
            sampleReward=0
            y=set(y) #因为y中有重复的药物 因此这里转变为集合 去掉重复的药物
            #得到初始的状态
            x=x.cuda(1)
            f=agent.reset(x)
            selectedAction =[]
            state = torch.cat((f, g), 1)
            for step in range(len(y)):
                #根据state选择action
                action=agent.act(state)
                #执行该action 得到reward 并更新状态
                reward,g=agent.step(action,selectedAction)
                #print('epoch:%d, action:%d, reward:%d'%(e,action,reward))
                next_state = torch.cat((f, g), 1)
                #将选择的action加入到selectedAction中
                selectedAction.append(int(action))
                sampleReward+=int(reward)
                #将经验放入经验池
                # 记忆这次transition
                agent.remember(state, action, reward, next_state)
            # 用之前的经验训练agent
            if len(agent.memory) >BATCH_SIZE:
                epochLoss.append(agent.replay(BATCH_SIZE))
            del sampleReward,x,y,f,state,g,action,reward,next_state,selectedAction
        if e%1==0:
            #每10轮评估一下模型在测试集上和验证集上的表现

            avg_reward, avg_jaccard, avg_recall,ddiNum=evl_val.evaluate()
            #将结果写入到文件中
            writer.writerow([e,float(sum(epochLoss)*1.0/len(epochLoss)),avg_reward,avg_jaccard,avg_recall,ddiNum])
            del avg_reward,avg_jaccard,avg_jaccard,ddiNum,epochLoss

    #画图
    #evl_val.plot_result(total_jaccard=total_jaccard,total_recall=total_recall,total_reward=total_reward)








