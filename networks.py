import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ActorNetwork(nn.Module):
  def __init__(self,obs_dim,act_dim):
    super().__init__()
    self.fc1 = nn.Linear(obs_dim,16)
    self.fc2 = nn.Linear(16,16)
    self.fc3 = nn.Linear(16,16)
    self.action = nn.Linear(16,act_dim)

    self.fc1.bias.data.fill_(0)
    self.fc2.bias.data.fill_(0)
    self.fc3.bias.data.fill_(0)
    self.action.bias.data.fill_(0)

    gain = nn.init.calculate_gain('relu')
    nn.init.xavier_uniform(self.fc1.weight,gain=gain)
    nn.init.xavier_uniform(self.fc2.weight,gain=gain)
    nn.init.xavier_uniform(self.fc3.weight,gain=gain)
    nn.init.xavier_uniform(self.action.weight,gain=gain)

  def _fwd(self,obs):
    x = F.relu(self.fc1(obs))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return x

  def get_action(self,obs):
    x = self._fwd(obs)
    x = self.action(x)
    probs = F.softmax(x,dim=-1)
    action = probs.multinomial()
    return action

  def evaluate_actions(self,obs,action):
    x = self._fwd(obs)
    x = self.action(x)
    log_probs = F.log_softmax(x,dim=-1)
    action_log_probs = log_probs.gather(1,action.unsqueeze(1))
    return action_log_probs

class CriticNetwork(nn.Module):
  def __init__(self,obs_dim):
    super().__init__()
    self.fc1 = nn.Linear(obs_dim,128)
    self.fc2 = nn.Linear(128,64)
    self.value = nn.Linear(64,1)

    self.fc1.bias.data.fill_(0)
    self.fc2.bias.data.fill_(0)
    self.value.bias.data.fill_(0)

    gain = nn.init.calculate_gain('relu')
    nn.init.xavier_uniform(self.fc1.weight,gain=gain)
    nn.init.xavier_uniform(self.fc2.weight,gain=gain)
    nn.init.xavier_uniform(self.value.weight,gain=gain)

  def forward(self,obs):
    x = F.relu(self.fc1(obs))
    x = F.relu(self.fc2(x))
    x = self.value(x)
    return x
"""
class ActorCriticNetwork(nn.Module):
  def __init__(self,obs_dim,act_dim):
    super().__init__()
    self.fc1 = nn.Linear(obs_dim,16)
    self.fc2 = nn.Linear(16,16)
    self.fc3 = nn.Linear(16,16)
    self.action = nn.Linear(16,act_dim)
    self.value = nn.Linear(16,1)

    self.fc1.bias.data.fill_(0)
    self.fc2.bias.data.fill_(0)
    self.fc3.bias.data.fill_(0)
    self.action.bias.data.fill_(0)
    self.value.bias.data.fill_(0)

    gain = nn.init.calculate_gain('relu')
    nn.init.xavier_uniform(self.fc1.weight,gain=gain)
    nn.init.xavier_uniform(self.fc2.weight,gain=gain)
    nn.init.xavier_uniform(self.fc3.weight,gain=gain)
    nn.init.xavier_uniform(self.action.weight,gain=gain)
    nn.init.xavier_uniform(self.value.weight,gain=gain)

  def _fwd(self,obs):
    x = F.relu(self.fc1(obs))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return x

  def get_action(self,obs):
    x = self._fwd(obs)
    act = self.action(x)
    probs = F.softmax(act,dim=-1)
    #action = probs.max(1,keepdim=True)[1]
    action = probs.multinomial()
    value = self.value(x)
    return value,action

  def evaluate_actions(self,obs,action):
    x = self._fwd(obs)
    act = self.action(x)
    log_probs = F.log_softmax(act,dim=-1)
    action_log_probs = log_probs.gather(1,action.unsqueeze(1))
    value = self.value(x)
    return value,action_log_probs
"""