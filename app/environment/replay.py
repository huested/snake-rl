from collections import namedtuple, deque
import random



class ExperienceReplay(object):

   # Initialize ExperienceReplay class 
   def __init__(self, D_size, sample_size):
      self.E = namedtuple('E', ('state_t0', 'action', 'state_t1', 'reward', 'done'))
      self.D_size = D_size
      self.experience = deque(maxlen=D_size)
      self.sample_size = sample_size

   # Save an experience
   def save(self, *args):
      self.experience.append(self.E(*args))

   # Randomly sample from experiences
   def sample(self):
      return random.sample(self.experience, self.sample_size)