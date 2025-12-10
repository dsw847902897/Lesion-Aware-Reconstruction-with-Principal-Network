from numpy import mean
from torch import tensor
import torch.utils.data as data
import os
import matplotlib.pyplot as plt

from Config import _C
from ActorCritic import ActorCritic
from env import SelectionEnv
from data import vocdataset

if __name__ == "__main__":
    env = SelectionEnv()
    agent = ActorCritic()
    rewards=[]
    for j in range(_C.EPOCH):
        params = {'batch_size': _C.BATCH_SIZE, 'shuffle': False, 'num_workers': _C.DATASET_NUM_WORKERS, 'pin_memory': _C.DATASET_PIN_MEMORY}
        train_set = vocdataset()
        train_loader = data.DataLoader(train_set, **params)
        returns = []
        i=0
        for X, y in train_loader:
            transitions = []
            # record episode's return to plot
            episode_return = 0
            state = env.reset(X.detach())
            while True:
                #env.render()
                action = agent.take_action(state)
                next_state, reward, done,_ = env.step(action)

                episode_return += reward
                transitions.append(
                    (tensor(state), tensor(action), tensor(next_state), tensor(reward), done))
                if done:
                    returns.append(episode_return)
                    break
                state = next_state
            agent.update(transitions)
            if (i + 1) % 200 == 0:
                print("episodes:{}->{}, episode_returns_mean:{}.".format(i - 199, i, mean(returns[i - 199:i])))
                rewards.append(mean(returns[i - 199:i]))
            i=i+1

        valid_set = vocdataset(is_train=False)
        valid_loader = data.DataLoader(valid_set, **params)
        val_returns=[]
        i=0
        for X, y in valid_loader:
            # record episode's return to plot
            episode_return = 0
            state = env.reset(X.detach())
            while True:
                action = agent.take_action(state)
                next_state, reward, done,_ = env.step(action)
                episode_return += reward
                if done:
                    val_returns.append(episode_return)
                    break
                state = next_state
        print("---------------------------------episodes:{}, test_returns_mean:{}.".format(j, mean(val_returns[:])))
        agent.save_ac(os.path.join(_C.CURRENT_PROJECT_DIR,'trained'),j)

    env.close()
    plt.plot(rewards)
    plt.title('train reward')
    plt.xlabel('iter')
    plt.ylabel('reward')
    plt.savefig('train_reward.png')
    plt.show()
