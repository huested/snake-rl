from replay import ExperienceReplay
from dqnet import DQNModel
import numpy as np
#import gym
import random
import torch
import matplotlib.pyplot as plt
from snake_env import SnakeEnvironment
from snake_learner import SnakeLearner

class SnakeTrainer():
   def __init__(self
         ,grid_height
         ,grid_width
         ,l2_size
         ,l3_size
         ,episodes
         ,steps_before_decaying
         ,steps_before_learning
         ,goal_rewards
         ,gamma=0.99
         ,learning_rate=0.001
         ,replay_size=100000
         ,replay_sample_size=128
         ,episodes_update_q_target=50
         ,update_q_every_n_time_steps=3
         ,epsilon_start=0.99
         ,epsilon_end=0.01):
      #Initialize variables
      self.grid_height=grid_height
      self.grid_width=grid_width
      self.l2size=l2_size
      self.l3size=l3_size
      self.episodes=episodes
      self.steps_before_decaying=steps_before_decaying
      self.steps_before_learning=steps_before_learning
      self.goal_rewards=goal_rewards
      self.gamma=gamma
      self.learning_rate=learning_rate
      self.replay_size=replay_size
      self.replay_sample_size=replay_sample_size
      self.episodes_update_q_target=episodes_update_q_target
      self.update_q_every_n_time_steps=update_q_every_n_time_steps
      self.epsilon_start=epsilon_start
      self.epsilon_cur=epsilon_start
      self.epsilon_end=epsilon_end
      self.episode=0
      self.iterations=0
      self.avg_100 = np.zeros(100)
      self.best_avg = None
      self.episode_decay_starts=None
      #Snake 
      self.env = SnakeEnvironment(height=self.grid_height, width=self.grid_width)
      self.snake =SnakeLearner(len(self.env.actions), self.env.width*self.env.height, replay_size, replay_sample_size, learning_rate, gamma, l2_size, l3_size)
   
   def DecayEpsilon(self):
      if self.iterations >= self.steps_before_decaying:
         self.epsilon_cur = self.epsilon_cur - (self.epsilon_start - self.epsilon_end)/(self.episodes-self.episode_decay_starts)


   def Train(self):
      self.env.reset()
      state_t1 = self.env.state

      # Iterate through episodes
      #decayed_epsilon = EPSILON
      #num_episodes = 0
      #rewards_per_episode = np.zeros(EPISODES)
      rolling_avg_array = np.zeros(self.episodes)

      for episode in range(self.episodes):
         self.episode=episode

         # Decay Epsilon (or dont)
         if self.iterations >= self.steps_before_decaying:
            self.episode_decay_starts = self.episode
            self.DecayEpsilon()
         
         # Reset environment
         tot_reward = 0
         state_t1 = self.env.reset()
         done = False 
         while done == False:         
            # Get s, a, s', r and save to replay memory
            state_t0 = state_t1
            action = self.snake.choose_action(state_t0.ravel(), self.epsilon_cur)
            state_t1, reward, done = self.env.step(action)
            tot_reward += reward
            self.snake.replay_memory.save(state_t0.ravel(), action, state_t1.ravel(), reward, done)

            # Update Q-policy if replay buffer is filled & every specified time steps
            if self.iterations > self.steps_before_learning and self.iterations %  self.update_q_every_n_time_steps == 0:
               self.snake.optimize(self.snake.replay_memory.sample())

            # Increment iterations
            self.iterations += 1

         

         # If it's time, swap parameters/weights of q_policy into q_target
         if self.episode % self.episodes_update_q_target:
            self.snake.swap()

         # Add rewards to array
         #rewards_per_episode[episode] = tot_reward

         # Store last 100 rewards
         self.avg_100[episode % 100] = tot_reward
         rolling_avg = np.average(self.avg_100)
         rolling_avg_array[episode] = rolling_avg
         
         if episode>=100:
            if self.best_avg==None:
               self.best_avg=rolling_avg
            else:
               self.best_avg=max(self.best_avg,rolling_avg)

         # Stop training if we're somehow doing well
         if rolling_avg> self.goal_rewards:
            print('Ending training b/c average reward is > ', self.goal_rewards)
            break

         # Increment episodes
         self.episode += 1

         # Print stuff
         #if self.episode % 100 == 0:
         #   print('episode ', episode, '\tavg rewards ', np.average(avg_100), '\tcum. iterations', iterations, '\tepsilon ', decayed_epsilon)

      # Save model
      #snake.save()

      return self.best_avg, rolling_avg

# Run and plot return per episode 
#Train()
