from collections import deque
from helper import plot
from model import Linear_QNet, QTrainer
import numpy as np
import random
from simulation import TempModel
import torch


# Constants for agent configuration
MAX_MEMORY = 100_000  # Maximum size of the experience replay memory
BATCH_SIZE = 1000  # Batch size for training
LR = 0.001  # Learning rate for Q-learning

class Agent:
    """An agent that learns from interactions with an environment using Q-learning.

    The agent maintains a memory of past experiences and uses them to update its
    Q-learning model. It also incorporates exploration and exploitation strategies
    to learn optimal policies.
    """

    def __init__(self):
        """Initializes the agent with a model, a trainer, and an empty memory."""
        self.n_train = 0  # Number of training iterations
        self.epsilon = 0  # Exploration-exploitation parameter
        self.gamma = 0.9  # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Memory for experience replay
        self.model = Linear_QNet()  # The Q-learning model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Trainer for the model

    def get_state(self, simulation):
        """Extracts the current state from the simulation.

        Args:
            simulation (TempModel): The simulation environment to extract state from.

        Returns:
            np.ndarray: An array representing the current state of the simulation.
        """
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
        """Stores an experience in the memory.

        Args:
            state (np.ndarray): The previous state.
            action (list): The action taken in that state.
            reward (float): The reward received for that action.
            next_state (np.ndarray): The resulting state after the action.
            done (bool): Whether the simulation has terminated.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Trains the Q-learning model with a batch of experiences from the memory.

        This method samples a batch of experiences and uses them to update the model
        through a QTrainer.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Random sample from memory
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Trains the model with a single step of experience.

        This method is used for online learning, where each experience is processed
        individually.

        Args:
            state (np.ndarray): The previous state.
            action (list): The action taken in that state.
            reward (float): The reward received for that action.
            next_state (np.ndarray): The resulting state after the action.
            done (bool): Whether the simulation has terminated.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Decides the action to take based on the current state.

        The action is chosen either randomly (exploration) or based on the model's
        prediction (exploitation), depending on the value of epsilon.

        Args:
            state (np.ndarray): The current state of the simulation.

        Returns:
            list: A list representing the action to take (one-hot encoded).
        """
        self.epsilon = 80 - self.n_train  # Update epsilon based on training count
        
        # 17 possible moves/actions
        final_move = [0] * 17  # Initialize the action list with zeros
        if random.randint(0, 200) < self.epsilon:  # Random exploration
            move = random.randint(0, 16)  # Randomly select a move
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
            prediction = self.model(state0)  # Get the model's prediction
            move = torch.argmax(prediction).item()  # Find the best action
            final_move[move] = 1  # Set the action in the final_move list

        return final_move


def train():
    plot_avgmseloss = []
    record = 20000  # Initial record value for MSE loss
    highavgmseloss = 20000  # Initial high value for MSE loss
    agent = Agent()  # Instantiate the agent
    simulation = TempModel()  # Instantiate the simulation environment
    while True:
        # Get old state
        state_old = agent.get_state(simulation)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, avgmseloss = simulation.play_step(final_move)
        if avgmseloss < highavgmseloss:
            highavgmseloss = avgmseloss
        state_new = agent.get_state(simulation)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # Train long memory, plot result
            simulation.reset()
            agent.n_train += 1
            agent.train_long_memory()

            # Update the model if a new record is set
            if highavgmseloss < record:
                record = highavgmseloss
                agent.model.save()  # Save the model if it's the best so far

            print('Training', agent.n_train, 'Record MSE Loss:', record)

        # Plot the MSE loss over time
        plot_avgmseloss.append(avgmseloss)
        plot(plot_avgmseloss)

if __name__ == '__main__':
    train()