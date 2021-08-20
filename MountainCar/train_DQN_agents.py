# implementation from https://github.com/gsurma/cartpole/blob/master/cartpole.py
import numpy as np
import gym
#from scores.score_logger import ScoreLogger # SHOUDL BE ADDED
from DQN import DQNSolver

def MountainCarTraining():
    env = gym.make("MountainCar-v0")
    #score_logger = ScoreLogger("MountainCar-v0")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space,action_space)
    run=0 #episode

    while True: # Will train forever?
        run+=1
        state = env.reset()
        state = np.reshape(state,[1,observation_space])
        step = 0
        while True:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward  # WHY?
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state,action,reward,state_next,terminal)
            state=state_next
            if terminal:
                print(" Run : "+str(run)+", exploration: "+str(dqn_solver.exploration_rate)+", score: "+ str(step)) # WHY ?
                #score_logger.add_score(step,run)
                break
            dqn_solver.experience_replay()

if __name__ == "__main__":
    MountainCarTraining()