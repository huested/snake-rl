from snake_trainer import SnakeTrainer

GAMMA = 0.99
LEARNING_RATE = 0.01
REPLAY_SIZE = 200000
REPLAY_SAMPLE_SIZE = 128
#EPISODES = 30000
EPISODES = 30000
EPISODES_UPDATE_Q_TARGET = 50
UPDATE_Q_EVERY_N_TIME_STEPS = 3
EPSILON = 0.99
EPSILON_END = 0.01
STEPS_BEFORE_DECAYING = 105000
STEPS_BEFORE_LEARNING = 100000
GOAL_REWARDS  = 30
L2SIZE=512
L3SIZE=64
GRID_HEIGHT=10
GRID_WIDTH=10


NUM_STATES=722

'''
trainer = SnakeTrainer(GRID_HEIGHT, GRID_WIDTH, L2SIZE, L3SIZE, EPISODES, STEPS_BEFORE_DECAYING, STEPS_BEFORE_LEARNING, GOAL_REWARDS)
best_100, last_100 = trainer.Train()
print("Best_100: ", int(best_100), "   Last_100: ", int(last_100))
'''

#l2s = [32, 64, 128, 256, 512, 1024, 2048, 5096, 10192]
#l3s = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 5096]
# l2s = [512, 1024, 2048, 5096, 10192]
# l3s = [64, 128, 256, 512, 1024]
l2s=[1024, 2048, 4096, 8192]
l3s=[128, 256, 512, 1024, 2048]
lrs = [0.001]
# l2s=[1024]
# l3s=[512]

#l2s=[2048, 4096, 8192]
#l3s=[64, 128, 256]
l2s=[128, 256, 512]
l3s=[8, 16, 64, 128, 256]
lrs = [0.001]

# l2s_l3s=[[128,64], [256, 64], [256, 128], [512, 128]]
# l2s_l3s=[[128,64], [256, 32], [128, 8], [64, 8], [512, 32]]
# l2s_l3s=[[128,64]]
# l2s_l3s=[[128,32], [256, 16], [256, 64], [512, 32], [1028, 128]]
l2s_l3s=[[128,64], [256, 16], [256, 8], [512, 16], [1028, 128], [1028,32], [1028, 16]]
l2s_l3s=[[128,128], [256, 128], [1028,64], [1028, 512]]
l2s_l3s=[[256,128], [128,128]]
l2s_l3s=[[128,128]]
l2s_l3s=[[256, 128], [1028,64], [1028, 512]]
l2s_l3s=[[64, 16], [128,8], [256, 128], [256, 32]]
l2s_l3s=[[512, 32], [512,64], [1028, 32], [256, 64], [1028, 64], [128,128], [256,128], [512,256]]
l2s_l3s=[[256, 64],[512,256]]
l2s_l3s=[[256, 64]]
gammas=[.99]
episodes=[80000]
episodes=[210000]
before_decay=[1000000]
replay_size = [1000000]
sample_size = [64]

'''
L2: 256, L3: 128, Gamma: 0.99, Eps: 40000, BD: 200000, Replay: 200000, Sample: 512
L2: 256, L3: 128, Gamma: 0.99, Eps: 40000, BD: 100000, Replay: 400000, Sample: 512
L2: 256, L3: 128, Gamma: 0.99, Eps: 40000, BD: 100000, Replay: 400000, Sample: 256
L2: 256, L3: 128, Gamma: 0.99, Eps: 40000, BD: 100000, Replay: 400000, Sample: 128
L2: 256, L3: 128, Gamma: 0.99, Eps: 40000, BD: 100000, Replay: 300000, Sample: 64

(512, 32, 0.99, 50000, 400000, 400000, 64)  =  (12.99, 12.77)
(512, 64, 0.99, 50000, 400000, 400000, 64)  =  (12.0, 11.99)
(1028, 32, 0.99, 50000, 400000, 400000, 64)  =  (6.18, 5.96)
(256, 64, 0.99, 50000, 400000, 400000, 64)  =  (13.6, 13.47)
(1028, 64, 0.99, 50000, 400000, 400000, 64)  =  (11.82, 11.4)
(128, 128, 0.99, 50000, 400000, 400000, 64)  =  (11.73, 11.66)
(256, 128, 0.99, 50000, 400000, 400000, 64)  =  (12.86, 12.73)
(512, 256, 0.99, 50000, 400000, 400000, 64)  =  (13.92, 12.84)
(256, 64, 0.99, 80000, 400000, 400000, 64)  =  (13.96, 12.39)
(512, 256, 0.99, 80000, 400000, 400000, 64)  =  (13.99, 13.72)

(256, 64, 0.99, 290000, 1000000, 1000000, 64)  =  (18.81, 17.87)

'''
def test():
   results = {}
   for l2, l3 in l2s_l3s:
      for lr in lrs:
         for g in gammas:
            for e in episodes:
               for bd in before_decay:
                  for r in replay_size:
                     for s in sample_size:
                        print(f"L2: {l2}, L3: {l3}, Gamma: {g}, Eps: {e}, BD: {bd}, Replay: {r}, Sample: {s}")
                        trainer = SnakeTrainer(GRID_HEIGHT, GRID_WIDTH, l2, l3, e, bd, bd, GOAL_REWARDS, gamma=g, num_states=NUM_STATES, replay_size=r, replay_sample_size=REPLAY_SAMPLE_SIZE, learning_rate=lr, epsilon_start=EPSILON, epsilon_end=EPSILON_END)
                        best_100, last_100 = trainer.TrainRadar()
                        #trainer = SnakeTrainer(GRID_HEIGHT, GRID_WIDTH, l2, l3, EPISODES, STEPS_BEFORE_DECAYING, STEPS_BEFORE_LEARNING, GOAL_REWARDS, num_states=None)
                        #best_100, last_100 = trainer.Train()
                        results[(l2, l3, g, e, bd, r, s)] = (round(best_100,2), round(last_100,2))
                        print(round(best_100, 2), "\t", round(last_100, 2))

   print(results)
   for key,value in results.items():
      print(key, " = ", value)

test()
         