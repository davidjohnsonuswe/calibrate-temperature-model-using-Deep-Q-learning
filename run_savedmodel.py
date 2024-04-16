import torch
from agent import Agent
from helper import plot
import random
from simulation import TempModel
from model import Linear_QNet

# Load the saved model
model = Linear_QNet()
model.load_state_dict(torch.load('model/model.pth'))

# Initialize the simulation
simulation = TempModel()
agent = Agent()
log_path = 'C:/Code/CC_ECC/target/components/internal/actuatorControl/units/acTempEst/test/log_action.txt'
epsilon = 80
while True:
    move = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    if random.randint(0, 200) < epsilon:
        action = random.randint(0, 16)
        move[action] = 1
    else:
        # Get the current state of the simulation
        state = agent.get_state(simulation)

        # Convert the state to a tensor
        state_tensor = torch.tensor(state, dtype=torch.float)
        # Get the Q-values for each action
        q_values = model(state_tensor)

        # Choose the action with the highest Q-value
        action = torch.argmax(q_values).item()
        
        # Convert the action to a direction
        move[action] = 1

        with open(log_path, "a+") as file:
            file.write(str(action))

    # Play the step and get the reward
    reward, done, avgmseloss = simulation.play_step(move)
    # plot_avgmseloss.append(avgmseloss)
    # plot(plot_avgmseloss, isvalidate=True)
    # If the simulation is over, break the loop
    if done:
        #break
        simulation.reset()
