import numpy as np
import random

'''
SnakeEnvironment
   Properties
      width: int, width of snake grid
      height: int, height of snake grid
      display: bool, whether to print state-action-reward each step
      n_food_range: tuple, min & max number of food spawned on snake grid
   
   Reset()
      Reset snake environment

   Step(action) => state,detailed state,reward,done
      Returns the next state given an action.
      Detailed state has three width x height arrays that separately represent the head, body, and food
      Done is a boolean indicating whether the snake is still going, or has collided with the wall/itself

'''
class SnakeEnvironment():
   def __init__(self, width=10, height=10, id=1, display=False, start_coord=None, start_length=None, n_food=1, n_food_range=(1,1), rand_start_length=True):
      self.id = id
      self.display = display
      self.start_length=start_length
      self.start_coord=start_coord
      self.done=True
      self.non_occupied_spaces = None
      self.food_coord = None
      self.food_coords=[]
      self.action = None
      self.actions = ('up', 'left', 'down', 'right')
      self.action_chain=[]
      self.head_val = self.id*10
      self.body_val = self.id
      self.n_food = n_food
      self.head=None
      self.n_food_range = n_food_range
      self.rand_start_length=rand_start_length
      self.prev_action=None
      self.food_got=0

      #Box properties
      self.width=width
      self.height=height
      
      self.state = np.zeros((self.height, self.width))
      self.state_head_only=np.zeros((self.height, self.width))
      self.state_body_only=np.zeros((self.height, self.width))
      self.state_food_only=np.zeros((self.height, self.width))
      self.state_body_direction=np.zeros((self.height, self.width))
      self.state_body_up=np.zeros((self.height, self.width))
      self.state_body_down=np.zeros((self.height, self.width))
      self.state_body_left=np.zeros((self.height, self.width))
      self.state_body_right=np.zeros((self.height, self.width))
      self.valid_moves = np.zeros((self.height,self.width))
      self.state_radar=np.zeros(((self.height*2), (self.width*2)-1))
      self.state_food_dist=[None, None]
      self.state_food_direction=[None, None, None, None]
      

      #Reward map
      self.reward_map = {
         'wall': -10.0,
         'food':1.0,
         'body':-10.0,
         'opponent_head':-10.0,
         'opponent_body':-10.0,
         'closer':0.25,
         'further':0.25,
         'move':-0.01
      }

      #Action map
      self.action_map = {
         'up':0,
         'down':2,
         'left':1,
         'right':3
      }
      self.action_map_opposite = {
         'up':2,
         'down':0,
         'left':3,
         'right':1
      }
      self.action_to_inverse_action = {
         0:2,
         1:3,
         2:0,
         3:1
      }
      self.action_to_body_direction = {
         0:self.state_body_up,
         1:self.state_body_left,
         2:self.state_body_down,
         3:self.state_body_right
      }
      self.action_to_delta_height = {
         0:-1,
         1:0,
         2:1,
         3:0
      }
      self.action_to_delta_width = {
         0:0,
         1:-1,
         2:0,
         3:1
      }


   def get_random_available_space_index(self):
      return random.randrange(len(self.non_occupied_spaces))

   def reset(self):
      self.done=False
      self.food_got=0
      self.state = np.zeros((self.height, self.width))
      self.state_head_only=np.zeros((self.height, self.width))
      self.state_body_only=np.zeros((self.height, self.width))
      self.state_body_direction=np.zeros((self.height, self.width))
      self.state_food_only=np.zeros((self.height, self.width))
      self.non_occupied_spaces = [(i,j) for i in range(self.height) for j in range(self.width)]
      self.action_chain=[]
      self.state_body_up=np.zeros((self.height, self.width))
      self.state_body_down=np.zeros((self.height, self.width))
      self.state_body_left=np.zeros((self.height, self.width))
      self.state_body_right=np.zeros((self.height, self.width))
      self.state_radar=np.zeros(((self.height*2), (self.width*2)-1))
         #last row will be used for food state

      #Food
      #for _ in range(self.n_food):
      self.food_coords=[]
      # for _ in range(random.randrange(self.n_food_range[0]-1, self.n_food_range[1])):
      #    rand_available_space = self.get_random_available_space_index()
      #    new_food = self.non_occupied_spaces.pop(rand_available_space)
      #    self.food_coords.append(new_food)
      #    self.state[new_food] = -1.0
      #    self.state_food_only[new_food] = 1.0

      rand_available_space = self.get_random_available_space_index()
      self.food_coord = self.non_occupied_spaces.pop(rand_available_space)
      self.state[self.food_coord] = -1.0
      self.state_food_only[self.food_coord] = 1.0

      #Snake head
      rand_available_space = self.get_random_available_space_index()
      self.head = self.non_occupied_spaces.pop(rand_available_space)
      self.state[self.head] = self.head_val
      self.state_head_only[self.head]=1.0

      #Food to head relationship
      self.set_head_to_food_distance()
      self.set_head_to_food_direction()

      #Snake body
      self.body = []
      if self.rand_start_length:
         prev_xy = self.head
         for _ in range(random.randrange(0,self.width)):
            rand_action = random.randrange(0,4)
            rand_action_opposite = self.action_to_inverse_action[rand_action]
            if self.available_move(prev_xy, rand_action):        
               new_xy = self.coord_plus_action(prev_xy, rand_action)
               self.body_add(new_xy, rand_action_opposite, add_to_front=True)
               prev_xy = new_xy

      #Direction
      self.action = None
      self.prev_action=None

      #Valid moves
      self.set_valid_moves_array()

      #Set radar; 
      #      1 1 1 1 1        1 1 1 1 1
      #      1 1 1 1 1        1 0 0 0 1
      #      1 0 0 0 1   ==>  1 0 0 0 1
      #      1 0 0 0 1        1 0 0 0 1
      #      1 0 0 0 1        1 1 1 1 1
      #
      #
      #      0 x 0            0 0 0
      #      0 0 0       ==>  0 x 0
      #      0 0 0            0 0 0
      #Todo: Second snake
      self.set_radar()
      

      if self.display:
         print(self.state)
      
      return self.state

   def set_radar(self):
      cur_x = self.head[1]
      left_offset_x = self.width-cur_x-1
      right_offset_x = (cur_x*-1)
      self.state_radar[:, 0:left_offset_x] = 1
      if right_offset_x <0:
         self.state_radar[:, right_offset_x:] = 1
      else:
         right_offset_x = self.width*2

      cur_y = self.head[0]
      upper_offset_y = self.height-cur_y-1
      bottom_offset_y = (cur_y*-1)-1
      self.state_radar[0:upper_offset_y , :] = 1
      if bottom_offset_y<1:
         self.state_radar[bottom_offset_y: , :] = 1
      else:
         bottom_offset_y = self.height*2

      self.state_food_radar = np.zeros(((self.height*2), (self.width*2)-1))
      self.state_food_radar[self.food_coord]=1
      self.state_radar[upper_offset_y:bottom_offset_y, left_offset_x:right_offset_x] = self.state_body_only
      self.set_last_row_of_radar_to_food_info()

      self.state_radar_all = np.asarray([self.state_food_radar, self.state_radar])
      #print(self.state_radar)
      return
   
   def set_head_to_food_distance(self):
      self.state_food_dist = [self.head[0]-self.food_coord[0],self.head[1]-self.food_coord[1]]
      return

   def set_head_to_food_direction(self):
      down=up=left=right=0
      if self.food_coord[0]>self.head[0]:
         down=1
      if self.food_coord[0]<self.head[0]:
         up=1
      if self.food_coord[1]>self.head[1]:
         right=1
      if self.food_coord[1]<self.head[1]:
         left=1
      self.state_food_direction=[up,left,down,right]

   def set_last_row_of_radar_to_food_info(self):
      self.state_radar[self.height*2-1, 0:4]=self.state_food_direction
      self.state_radar[self.height*2-1, 4:6]=self.state_food_dist
      return

   def coord_plus_action(self, coord, action):
      return (coord[0]+self.action_to_delta_height[action],coord[1]+self.action_to_delta_width[action])

   def available_move(self,coord,action):
      new_coord = self.coord_plus_action(coord, action)
      if new_coord in self.non_occupied_spaces:
         return True
      else:
         return False

   def update_body_direction(self, coord, action, new_value):
      if action==0:
         self.state_body_up[coord] = new_value
      elif action==1:
         self.state_body_left[coord] = new_value
      elif action==2:
         self.state_body_down[coord] = new_value
      elif action==3:
         self.state_body_right[coord] = new_value
      return
   
   def body_add(self, coord, action, add_to_front=False):
      new_body = self.non_occupied_spaces.pop(self.non_occupied_spaces.index(coord))
      #print("body_add() ", new_body)
      if add_to_front:
         self.body.insert(0, new_body)
         self.action_chain.insert(0,action)
      else:
         self.body.append(new_body)
         self.action_chain.append(action)

      self.state[new_body] = self.body_val
      self.state_body_only[new_body]=1.0
      self.state_body_direction[new_body]=action+1
      #self.action_to_body_direction[action][new_body]=1.0
      self.update_body_direction(new_body, action, 1.0)
      return

   def body_pop_tail(self):
      old_tail = self.body.pop(0)
      old_tail_action = self.action_chain.pop(0)
      self.non_occupied_spaces.append(old_tail)
      #self.action_to_body_direction[old_tail_action][old_tail]=0.0
      self.update_body_direction(old_tail, old_tail_action, 0.0)
      self.state[old_tail]=0.0
      self.state_body_only[old_tail]=0.0
      self.state_body_direction[old_tail]=0.0
      return
   
   def head_move(self, old_coord, new_coord):
      self.head = self.non_occupied_spaces.pop(self.non_occupied_spaces.index(new_coord))
      self.non_occupied_spaces.append(old_coord)
      self.state_head_only[old_coord]=0.0
      self.state_head_only[new_coord]=1.0
      self.state[old_coord]=0.0
      self.state[new_coord]=self.head_val
      return
   
   def food_remove(self, coord):
      self.state_food_only[coord]=0.0
      self.non_occupied_spaces.append(coord)
      return
   
   def food_spawn(self):
      rand_available_space = self.get_random_available_space_index()
      self.food_coord = self.non_occupied_spaces.pop(rand_available_space)
      #new_food = self.non_occupied_spaces.pop(rand_available_space)
      #self.food_coords.append(new_food)
      self.state[self.food_coord] = -1.0
      self.state_food_only[self.food_coord]=1.0
      return

   def set_valid_moves_array(self):
      if self.available_move(self.head, 0):
         self.valid_moves[0][0]=0.0
      else:
         self.valid_moves[0][0]=1.0
      if self.available_move(self.head,1):
         self.valid_moves[0][1]=0.0
      else:
         self.valid_moves[0][1]=1.0
      if self.available_move(self.head,2):
         self.valid_moves[0][2]=0.0
      else:
         self.valid_moves[0][2]=1.0
      if self.available_move(self.head,3):
         self.valid_moves[0][3]=0.0
      else:
         self.valid_moves[0][3]=1.0
      return
   
   def step(self, action):
      reward = 0.0

      #Cant move backwards into body
      if self.body != [] and self.body[-1]==(self.head[0]+self.action_to_delta_height[action], self.head[1]+self.action_to_delta_width[action]):
         if self.prev_action == None:
            #Edge case when starting with body length>0, and do not have a previous action
            self.prev_action = random.randrange(0,4)
            while(self.prev_action==action):
               self.prev_action = random.randrange(0,4)
         action=self.prev_action

      #Calculate new head coordinate
      new_head = (self.head[0]+self.action_to_delta_height[action], self.head[1]+self.action_to_delta_width[action])
      

      
      #Check if moved into wall, food, or self
      #Collision: Wall
      if new_head[0] < 0 or new_head[0] >= self.height \
         or new_head[1] < 0 or new_head[1] >= self.width:
         reward = self.reward_map['wall']
         self.done = True
      #Collision: Body
      elif new_head in self.body[1:]:
         reward = self.reward_map['body']
         self.done = True
      #No game ending collision
      else:
         #Food
         old_head = self.head
         if new_head == self.food_coord:
            #if new_head in self.food_coords:
               #self.food_coords.pop(self.food_coords.index(new_head))
            #self.state_food_only[new_head]=0.0
            reward=self.reward_map['food']
            self.food_remove(new_head)
            self.head_move(old_head, new_head)
            self.body_add(old_head, action)
            
            if len(self.non_occupied_spaces)>0:
               self.food_spawn()
            self.food_got+=1

         #No food
         else:
            #Calculate reward based on distance to food
            new_dist= abs(new_head[0]-self.food_coord[0]) + abs(new_head[1]-self.food_coord[1])
            old_dist= abs(self.head[0]-self.food_coord[0]) + abs(self.head[1]-self.food_coord[1])
            if new_dist > old_dist:
               reward = self.reward_map['further']
            else:
               reward = self.reward_map['closer']
            
            if self.body != []:
               self.body_pop_tail()
               self.head_move(old_head, new_head)
               self.body_add(old_head, action)
            else:
               self.head_move(old_head, new_head)
       
      #Store action
      self.prev_action = action

      #Get valid actions
      self.set_valid_moves_array()
      self.set_head_to_food_distance()
      self.set_head_to_food_direction()
      self.set_radar()
      
      #print(self.valid_moves)

      #Print
      if self.display:
         print(self.state)
         print('Action: ', self.actions[action], "; Reward: ", reward)

      detailed_state = np.asarray([self.state_head_only, self.state_food_only, self.state_body_up, self.state_body_down, self.state_body_left, self.state_body_right, self.valid_moves])
      return self.state, detailed_state, self.state_radar_all, reward, self.food_got, self.done