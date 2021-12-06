import torch 
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
   def __init__(self, num_actions, num_states, layer2_size=9216, layer3_size = 128):
      # Initialize neural network model -- 2 hidden linear layers
      super(DQNModel,self).__init__()
      self.ll1= nn.Linear(num_states,layer2_size)
      self.ll2 = nn.Linear(layer2_size,layer3_size)
      self.ll3 = nn.Linear(layer3_size,num_actions)
      self.seed = torch.manual_seed(123)

   def forward(self,state):
      # Forward propagation of state
      state = F.relu(self.ll1(state))
      state = F.relu(self.ll2(state))
      return self.ll3(state)