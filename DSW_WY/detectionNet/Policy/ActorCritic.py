from torch import tensor, log
from torch.distributions import Categorical
from torch.optim import Adam
import torch
import os

from Model.AC import PolicyNet, ValueNet
from Config import _C


class ActorCritic:
    def __init__(self, actor_lr=_C.ACTOR_LR, critic_lr=_C.CRITIC_LR, gamma=_C.GAMMA, device=_C.DEVICE):
        self.actor = PolicyNet().to(device)
        self.critic = ValueNet().to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma

        self.device = device
        self.count = 0

    def take_action(self, state):
        probs = self.actor(tensor(state).float().to(self.device))
        categorical = Categorical(probs)
        return categorical.sample().item()

    def update(self, transitions):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        for state, action, next_state, reward, done in transitions:
            state = state.float().to(self.device)
            td_target = reward.to(self.device) + self.gamma * self.critic(next_state.float().to(self.device)) * (1 - done)
            critic_loss = pow(td_target.detach() - self.critic(state), 2)

            delta = td_target - self.critic(state)

            actor_loss = -log(self.actor(state)[action]) * delta.detach()
            actor_loss.backward()
            critic_loss.backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()
    
    def save_ac(self,save_dir,epoch_id):
        torch.save(self.actor.state_dict(),os.path.join(save_dir,f'actor_epoch{epoch_id}.pth'))
        torch.save(self.critic.state_dict(),os.path.join(save_dir,f'critic_epoch{epoch_id}.pth'))
        print(f'model AC in epoch: {epoch_id} has been saved!')

