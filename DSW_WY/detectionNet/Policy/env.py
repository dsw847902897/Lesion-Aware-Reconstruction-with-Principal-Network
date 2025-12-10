import gym
from gym import spaces
import numpy as np
import torch
import torch.utils.data as data

from Config import _C
from Model.AC import encodercnn
from Model.mobile import mobile
from data import vocdataset

class SelectionEnv(gym.Env):
    def __init__(self):
        super(SelectionEnv, self).__init__()
        self.device=_C.DEVICE
        self.CNNEncoder = encodercnn()
        self.CNNEncoder.eval()
        self.RewardNet = mobile()
        self.RewardNet.eval()
        self.action_space = spaces.Discrete(9)  # 9个动作选项
        self.observation_space = spaces.Box(low=-50, high=50, shape=(9,))

    def reset(self,X):
        X = X.to(self.device)
        output = self.CNNEncoder(X)
        output=output.detach().cpu().numpy().tolist()
        
        self.state_raw=[0,0,0,0,0,0,0,0,0]
        self.state=[0,0,0,0,0,0,0,0,0]
        for i in range(0,9):
            state_list=output[i]
            sorted_list = sorted(enumerate(state_list), key=lambda x: x[1], reverse=True)
            _, max_element = sorted_list[0]
            _, second_max_element = sorted_list[1]
            value_diff=max_element - second_max_element
            self.state_raw[i]=value_diff

        reward_output=self.RewardNet(X)
        reward_output=reward_output.detach().cpu().numpy().tolist()
        self.out_rewards=[0,0,0,0,0,0,0,0,0]
        for i in range(0,9):
            r_l=reward_output[i]
            sorted_rl=sorted(enumerate(r_l), key=lambda x: x[1], reverse=True)
            _, max_confidence = sorted_rl[0]
            _, second_max_confidence = sorted_rl[1]
            rew=max_confidence-second_max_confidence
            self.out_rewards[i]=rew
            
        #归一化state
        min_val = min(self.state_raw)
        max_val = max(self.state_raw)
        state = [(x - min_val) / (max_val - min_val) for x in self.state_raw]
        max_i = np.argmax(state)
        for i in range(0,9):
            if i != max_i:
                self.state[i]=state[i]-state[max_i]
            else:
                self.state[i]=state[i]

        return self.state

    def step(self, action):
        # 检查选择的是否是苹果
        refer_value=self.out_rewards[4]
        rewards=[0,0,0,0,0,0,0,0,0]
        for i in range(0,len(self.out_rewards)):
            rewards[i]=self.out_rewards[i]-refer_value
        reward = rewards[action]
        done = True  # 一次选择后游戏结束

        return self.state, reward, done, {}
    

if __name__=='__main__':
    params = {'batch_size': _C.POINT_CHOOSE_BS, 'shuffle': False, 'num_workers': _C.DATASET_NUM_WORKERS, 'pin_memory': _C.DATASET_PIN_MEMORY}
    valid_set = vocdataset()
    valid_loader = data.DataLoader(valid_set, **params)
    with torch.no_grad():
        for X, y in valid_loader:
            f=SelectionEnv()
            s=f.reset(X,y)
            print(s)
            f.step(1)

