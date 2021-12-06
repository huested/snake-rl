import numpy as np
from dqnet import DQNModel
from replay import ExperienceReplay
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class SnakeLearner():
   def __init__(self, num_actions, num_states, replay_size, sample_size, learning_rate, gamma, l2size=9216, l3size=128):
      # Device
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.num_actions = num_actions
      self.num_states = num_states
      # Initialize networks
      self.q_policy = DQNModel(self.num_actions, self.num_states, layer2_size=l2size,layer3_size=l3size).to(self.device)
      self.q_target = DQNModel(self.num_actions, self.num_states, layer2_size=l2size,layer3_size=l3size).to(self.device)
      # Optimizer: https://pytorch.org/docs/stable/optim.html
         # Adam seems like fastest option?
      self.optimizer = optim.Adam(self.q_policy.parameters(), lr=learning_rate)
      # Initialize replay memory
      self.replay_memory = ExperienceReplay(replay_size, sample_size)
      # Gamma 
      self.gamma = gamma
      

   # INPUT: array of state
   # OUTPUT: action int
   def choose_action(self, state, epsilon):
      # Epsilon greedy
      randnum = random.random()
      self.q_policy.eval()
      if randnum > epsilon:
         # Calculate max action based on q policy
         with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            max_action = torch.argmax(self.q_policy(state_tensor)).item()
            return max_action
      else:
         # Return random action
         return random.randrange(self.num_actions)

   #sample batch from memory, calculate loss, backprop
   def optimize(self, experience):
      # Set learn/eval for Qs
      self.q_policy.train()
      self.q_target.eval()

      # Turn sampled experiences to tensors
      es = self.replay_memory.E(*zip(*experience))
      es_state_t0 = np.vstack([e for e in es.state_t0])
      es_state_t1 = np.vstack([e for e in es.state_t1])
      es_reward = np.vstack([e for e in es.reward])
      es_action = np.vstack([e for e in es.action])
      es_done = np.vstack([e for e in es.done]).astype(np.uint8)

      tensor_es_state_t0 = torch.from_numpy(es_state_t0).float().to(self.device)
      tensor_es_state_t1 = torch.from_numpy(es_state_t1).float().to(self.device)
      tensor_es_reward = torch.from_numpy(es_reward).float().to(self.device)
      tensor_es_action = torch.from_numpy(es_action).long().to(self.device)
      tensor_es_done= torch.from_numpy(es_done).long().to(self.device)

      # Q(s) --> a_1...a_n values 
      q_s_a = self.q_policy(tensor_es_state_t0).gather(1, tensor_es_action)

      # Q(s, a) values -- Q(S,a;w)
      #q_s_a = q_s.gather(1, tensor_es_action)

      # Expected return under target policy -- r + gamma*Q(s',a';w-)
      with torch.no_grad():
         state_t1_v = self.q_target(tensor_es_state_t1).max(1)[0].detach().unsqueeze(1).to(self.device)
      r_gamma_state_t1_v = tensor_es_reward + (self.gamma*state_t1_v)*(1-tensor_es_done)

      # Calculate loss and backpropogate
      #print('policy ', q_s_a)
      #print('target ', r_gamma_state_t1_v)
      loss = F.smooth_l1_loss(q_s_a, r_gamma_state_t1_v).to(self.device) # https://pytorch.org/docs/stable/nn.functional.html
      #print('loss', loss)
      self.optimizer.zero_grad()
      loss.backward()

      # Clip gradient to deal with runaway values
      for parameter in self.q_policy.parameters():
        parameter.grad.data.clamp_(-1, 1)
      self.optimizer.step()

   # Swap parameters/weights of q_policy into q_target
   def swap(self):
      self.q_target.load_state_dict(self.q_policy.state_dict())

   # Save model to file
   def save(self, filepath):
      torch.save(self.q_policy.state_dict(), filepath)

