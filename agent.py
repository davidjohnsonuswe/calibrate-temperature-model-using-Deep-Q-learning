from collections import deque
from helper import plot
from model import Linear_QNet, QTrainer
import numpy as np
import random
from simulation import TempModel
import torch


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_train = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, simulation):

        currentloss = simulation.read_currentloss()
        isterminated = simulation.is_termination()
        state = [
            # Simulation is terminated
            isterminated,
            #
            simulation.paramsdict['AlfaCplgOilToPmpHd'] < 1,
            simulation.paramsdict['AlfaCplgOilToLmlLo'] < 6,
            simulation.paramsdict['AlfaCplgOilToLmlHi'] < 11,
            simulation.paramsdict['AlfaCplgOilToCplg'] < 11,
            simulation.paramsdict['AlfaCplgOilToFnGear'] < 1,
            simulation.paramsdict['AlfaCplgToPmpHd'] < 1,
            simulation.paramsdict['AlfaCplgToLml'] < 1,
            simulation.paramsdict['AlfaCplgToFinGear'] < 3,
            simulation.paramsdict['AirToPumpHead'] < 41,
            simulation.paramsdict['AirToCoupling'] < 26,
            simulation.paramsdict['AirToFinalGear'] < 26, 
            # Loss
            currentloss <= 10,
            currentloss > 10 and currentloss <= 20,
            currentloss > 20 and currentloss <= 30,
            currentloss > 30 and currentloss <= 50,
            currentloss > 50 and currentloss <= 100,
            currentloss > 100 and currentloss <= 200,
            currentloss > 200 and currentloss <= 300,
            currentloss > 300 and currentloss <= 400,
            currentloss > 400 and currentloss <= 500,
            currentloss > 500 and currentloss <= 600,
            currentloss > 600 and currentloss <= 700,
            currentloss > 700 and currentloss <= 800,
            currentloss > 800
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_train
        # 11 possible moves (11 parameters)
        final_move = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 16)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_avgmseloss = []
    record = 20000
    highavgmseloss = 20000
    agent = Agent()
    simulation = TempModel()
    while True:
        # get old state
        state_old = agent.get_state(simulation)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, avgmseloss = simulation.play_step(final_move)
        if avgmseloss < highavgmseloss:
            highavgmseloss = avgmseloss
        state_new = agent.get_state(simulation)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory, plot result
            simulation.reset()
            agent.n_train += 1
            agent.train_long_memory()

            if highavgmseloss < record:
                record = highavgmseloss
                agent.model.save()

            print('Training', agent.n_train, 'Record MSE Loss:', record)

        plot_avgmseloss.append(avgmseloss)
        plot(plot_avgmseloss)

if __name__ == '__main__':
    train()