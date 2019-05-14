#!/usr/bin/env python

import numpy as np
import time
import rospy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from robot_sim.msg import RobotState
from robot_sim.srv import RobotAction
from robot_sim.srv import RobotActionRequest
from robot_sim.srv import RobotActionResponse
from robot_sim.srv import RobotPolicy
from robot_sim.srv import RobotPolicyRequest
from robot_sim.srv import RobotPolicyResponse

class ReplayMemory(object):
    def __init__(self):
        self.capacity = 800
        self.memory = []
    
    def push(self,trans):
        if len(self.memory) < self.capacity:
            self.memory.append(trans)
        else:
            self.memory.pop(0)
            self.memory.append(trans)

    def sample(self,batch_size):
        return np.array(random.sample(self.memory,batch_size))

    def len(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,2)
    
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

class Infra(object):
    def __init__(self):
        self.batch_size = 32
        self.gamma = 0.7
        self.loss = 1
        self.act_network = DQN()
        self.target_network = DQN()
        self.action_space = [-10,10]
        self.replaymemory = ReplayMemory()
        self.optimizer = optim.RMSprop(self.act_network.parameters())
        self.cartpole_action_service = rospy.ServiceProxy('cartpole_robot',RobotAction)
        self.policy_service = rospy.Service('cartpole_policy', RobotPolicy, self.robot_policy_service)
    
    def robot_policy_service(self, req):
        response = RobotPolicyResponse()
        state = torch.tensor(req.robot_state)
        index = self.act_network(torch.tensor(state)).max(0)[1]
        action = self.action_space[index]
        response.action = action
        return response

    def reset_robot(self, state):
        req = RobotActionRequest()
        req.reset_robot = True
        req.reset_pole_angle = state[1,0]
        response = self.cartpole_action_service(req)
        return response
    
    def learn(self):
        episode_reward = 0
        total_episodes = 512
        for episode in range(1,total_episodes + 1):
            episode_reward = 0
            s_1 = np.zeros((4,1))
            s_1[1,0] = np.random.uniform(-3,3)
            response = self.reset_robot(s_1)
            print(response.robot_state)
            epsilon = max(1 - float(episode)/total_episodes, 0.01)
            for t in range(1,1024):
                s_t = response.robot_state
                req = RobotActionRequest()
                req.reset_robot = False
                dice = np.random.uniform(0,1)
                if dice < epsilon:
                    a_t = np.random.choice([10,-10])
                else:
                    index = self.act_network(torch.tensor(s_t)).max(0)[1]
                    a_t = self.action_space[index]
                req.action = [a_t]
                response = self.cartpole_action_service(req)
                s_t_1 = response.robot_state
                x = s_t_1[0]
                theta = s_t_1[1]
                if abs(x) > 1.2:
                    break
                elif abs(theta) > 3:
                    r_t_1 = 0
                    s_t_1 = None
                    transition = (s_t, a_t, r_t_1, s_t_1)
                    self.replaymemory.push(transition)
                    break
                else:
                    r_t_1 = 1
                    episode_reward += r_t_1
                    transition = (s_t, a_t, r_t_1, s_t_1)
                    self.replaymemory.push(transition)
                if t % 2 == 0:
                    self.train()
                # print('state:',response.robot_state,'action:',req.action)
            print('episode =',episode, 'episode reward =',episode_reward)
            print('loss =',self.loss)
            # report episode reward
            if episode > 5 and episode % 2 == 0:
                # print('=========update target network========')
                self.target_network.load_state_dict(self.act_network.state_dict())
    
    def train(self):
        if self.replaymemory.len() < self.batch_size:
            return
        transitions = self.replaymemory.sample(self.batch_size)
        state_batch = torch.tensor(transitions[:,0].tolist())
        # print('state batch',state_batch)
        action_temp = transitions[:,1]
        action_batch = torch.tensor([0 if x == -10 else 1 for x in action_temp]).view(-1,1)
        print('action batch',action_batch)
        reward_batch = torch.tensor(transitions[:,2].tolist(),dtype = torch.float)
        # print(reward_batch)
        state_action_values = self.act_network(state_batch).gather(1,action_batch)
        print('act state batch',self.act_network(state_batch))
        print('state action values',state_action_values)
        next_state_batch = transitions[:,3]
        print('next state batch:',next_state_batch)
        next_state_values = []
        for i in range(self.batch_size):
            if next_state_batch[i] == None:
                next_state_values.append(0.0)
            else:
                value = self.target_network(torch.tensor(next_state_batch[i],dtype = torch.float)).max(0)[0]
                print(self.act_network(torch.tensor(next_state_batch[i],dtype = torch.float)),value)
                next_state_values.append(value)
        next_state_values = torch.tensor(next_state_values,dtype = torch.float)
        # print('next state values',next_state_values)
        expected_state_action_values = ((next_state_values * self.gamma) + reward_batch).view(-1,1)
        # print('expected state action values',expected_state_action_values)

        loss = F.smooth_l1_loss(state_action_values,expected_state_action_values)
        self.loss = float(loss)
        # print('loss = ',loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.act_network.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        return


if __name__ == '__main__':
    rospy.init_node('learn_dqn', anonymous=True)
    np.random.seed(rospy.get_rostime().secs)
    learner = Infra()
    learner.learn()
    time.sleep(1)
    rospy.spin()