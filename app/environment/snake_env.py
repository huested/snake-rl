import numpy as np
import random

class SnakeEnvironment():
   def __init__(self, width=2, height=3, id=1, display=False, start_coord=None, start_length=None, n_food=1):
      self.id = id
      self.display = display
      self.start_length=start_length
      self.start_coord=start_coord
      self.done=True
      self.non_occupied_spaces = None
      self.food_coord = None
      self.action = None
      self.actions = ('up', 'down', 'left', 'right')
      self.head_val = self.id*10
      self.body_val = self.id
      self.n_food = n_food

      #Box properties
      self.width=width
      self.height=height
      self.state = np.zeros((self.height, self.width))

      #Reward map
      self.reward_map = {
         'wall': -50,
         'food':1,
         'body':-50,
         'opponent_head':-50,
         'opponent_body':-50
      }

   def get_random_available_space_index(self):
      return random.randrange(len(self.non_occupied_spaces))

   def reset(self):
      self.done=False
      self.state = np.zeros((self.height, self.width))
      self.non_occupied_spaces = [(i,j) for i in range(self.height) for j in range(self.width)]
      
      #Food
      #for _ in range(self.n_food):
      rand_available_space = self.get_random_available_space_index()
      self.food_coord = self.non_occupied_spaces.pop(rand_available_space)
      self.state[self.food_coord] = -1

      #Snake head
      rand_available_space = self.get_random_available_space_index()
      self.head = self.non_occupied_spaces.pop(rand_available_space)
      self.state[self.head] = self.head_val

      #Snake body
      self.body = []

      #Direction
      self.action = None

      #Todo: Second snake head
      #Todo: Start with snake with a body
      if self.display:
         print(self.state)
      
      return self.state


   def step(self, action):
      reward = 0

      #Cant move backwards into body
      if self.body != []:
         if action==0 and (self.head[0]-1, self.head[1]) == self.body[-1]:
            action = self.prev_action
         elif action==1 and (self.head[0]+1, self.head[1]) == self.body[-1]:
            action = self.prev_action
         elif action==2 and (self.head[0], self.head[1]-1) == self.body[-1]:
            action = self.prev_action
         elif action==3 and (self.head[0], self.head[1]+1) == self.body[-1]:
            action = self.prev_action

      #Move head and body
      #0: UP
      if action==0:
         new_head = (self.head[0]-1, self.head[1])
      #1: DOWN
      elif action==1:
         new_head = (self.head[0]+1, self.head[1])
      #2: LEFT
      elif action==2:
         new_head = (self.head[0], self.head[1]-1)
      #3: RIGHT
      elif action==3:
         new_head = (self.head[0], self.head[1]+1)
      else:
         new_head = None
         #error
      

      
      #Check if moved into wall, food, or self
      #Wall
      if new_head[0] < 0 or new_head[0] >= self.height \
         or new_head[1] < 0 or new_head[1] >= self.width:
         reward = self.reward_map['wall']
         self.done = True
      #Body
      elif new_head in self.body[1:]:
         reward = self.reward_map['body']
         self.done = True
      #No collision
      else:
         #Food
         if new_head == self.food_coord:
            reward=self.reward_map['food']
            self.body.append(self.head)
            self.state[self.head] = self.body_val
            self.head = new_head
            rand_available_space = self.get_random_available_space_index()
            self.food_coord = self.non_occupied_spaces.pop(rand_available_space)
            self.state[self.food_coord] = -1
         #Nothing
         else:
            if self.body != []:
               old_body = self.body.pop(0)
               self.non_occupied_spaces.append(old_body)
               self.state[old_body]=0
               self.body.append(self.head)
               self.state[self.head] = self.body_val
               self.head =self.non_occupied_spaces.pop(self.non_occupied_spaces.index(new_head))
            else:
               self.non_occupied_spaces.append(self.head)
               self.state[self.head] = 0
               self.head =self.non_occupied_spaces.pop(self.non_occupied_spaces.index(new_head))
         self.state[self.head] = self.head_val
       
      


      #Store action
      self.prev_action = action

      #Print
      if self.display:
         print(self.state)
         print('Action: ', self.actions[action], "; Reward: ", reward)

      return self.state, reward, self.done

   