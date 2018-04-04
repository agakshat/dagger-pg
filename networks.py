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
    nn.init.xavier_uniform(self.fc1.weight)
    nn.init.xavier_uniform(self.fc2.weight)
    nn.init.xavier_uniform(self.fc3.weight)
    nn.init.xavier_uniform(self.action.weight)

  def _fwd(self,obs):
    x = F.relu(self.fc1(obs))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return x

  def get_action(self,obs):
    x = self._fwd(obs)
    x = self.action(x)
    probs = F.softmax(x,dim=-1)
    #log_probs = F.log_softmax(x)
    #action = probs.max(1,keepdim=True)[1]
    return None,probs