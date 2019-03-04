# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/4 0004 上午 9:34
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用pytorch实现DQN框架
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(N_STATE,50)
        self.fc1.weight.data.normal_(0,0.1)

        self.out=nn.Linear(50,N_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        action_value=self.out(x)
        return action_value

class DQN(object):
    def __init__(self):
        self.eval_net,self.target_net=Net(),Net()
        self.learn_step_counter=0  # for target updating
        self.memory=np.zeros((MEMOEY_CAPACITY,N_STATE*2+2))
        self.memory_counter=0  # for storing memory
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()

    def choose_action(self,x):
        x=torch.unsqueeze(torch.FloatTensor(x),0)
        #input only one example
        if np.random.uniform()<EPSILON: # greedy
            action_value=self.eval_net.forward(x)
            action=torch.max(action_value,1)[1].data.numpy()
            action=action[0] if ENV_A_SHAPE==0 else action.reshpae(ENV_A_SHAPE)
        else:
            action=np.random.randint(0,N_ACTIONS)
            action=action if ENV_A_SHAPE==0 else action.reshape(ENV_A_SHAPE)
        return action
    def store_transition(self,s,a,r,s_):
        transition=np.hstack((s,[a,r],s_))
        #replace the old memory with new memory
        index=self.memory_counter%MEMOEY_CAPACITY
        self.memory[index,:]=transition
        self.memory_counter+=1

    def learn(self):
        if self.learn_step_counter%TARGET_REPLAY_ITER==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch transistions
        sample_index=np.random.choice(MEMOEY_CAPACITY,BATCH_SIZE)
        b_memory=self.memory[sample_index,:]
        b_s=torch.FloatTensor(b_memory[:,N_STATE])
        b_a=torch.LongTensor(b_memory[:,N_STATE:N_STATE+1].astype(int))
        b_r=torch.FloatTensor(b_memory[:,N_STATE+1:N_STATE+2])
        b_s=torch.FloatTensor(b_memory[:,-N_STATE:])

        #
        q_eval=self.eval_net(b_s).gather(1,b_a)
        q_next=self.target_net(b_s).detach()
        q_target=b_r+GAMMA*q_next.max(1)[0].view(BATCH_SIZE,1)
        loss=self.loss_func(q_eval,q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




if __name__ == '__main__':
    #超参数
    BATCH_SIZE=32
    LR=0.01
    EPSILON=0.9
    GAMMA=0.9
    TARGET_REPLAY_ITER=100
    MEMOEY_CAPACITY=2000
    env=gym.make('CartPole-v0')
    env=env.unwrapped
    N_ACTIONS=env.action_space.n
    N_STATE=env.observation_space.shape[0]
    #????
    ENV_A_SHAPE=0 if isinstance(env.action_space.sample(),int) else env.action_space.sample().shape

    dqn=DQN()
    print('\n Collecting experience...')
    for i_episode in range(400):
        s=env.reset()
        ep_r=0
        while True:
            env.render()
            a=dqn.choose_action(s)

            #take action
            s_,r,done,info=env.step(a)
            #modify the reward
            x,x_dot,theta,theta_dot=s_
            r1=(env.x_threshold-abs(x))/env.x_threshold-0.8
            r2=(env.theta_threshold_radians-abs(theta))/env.theta_threshold_radians-0.5
            r=r1+r2

            dqn.store_transition(s,a,r,s_)
            ep_r += r
            if dqn.memory_counter > MEMOEY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                break
            s = s_