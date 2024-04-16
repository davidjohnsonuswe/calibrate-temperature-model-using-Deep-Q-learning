# Calibrate Temperature model using Deep Q-learning

While Q-Learning can be very effective, it has a major limitation in that it can only handle environments with small state and action spaces, as it requires more memory and time to create and store the Q-table.
Deep Q-network (DQN) addresses this limitation by using deep neural networks (DNN) to approximate the Q-value function. DQN takes in the state as input and outputs the Q-value for each action. The weights of the network are then updated to minimize the difference between the predicted Q-values and the target Q-values.

Deep Q-learning uses the experience replay technique where the past transition at each time-step is stored in a replay memory. During training, mini-batches of transitions are randomly sampled from this replay memory instead of using just the latest transition. This approach ensures a diverse and uncorrelated set of experiences for learning, thereby improving the stability and efficiency of the learning process.
In addition, an greedy policy is used to selects and executes an action to ensure good coverage of the state and action space. Finally, through backpropagation, the weights of the main DNN are updated to minimize the loss, thus improving the accuracy of Q-value estimation.

This project use Deep Q-learning to calibrate the temperature Matlab model.

## Structure

1. agent.py: The implementation of the agent that interacts with the environment, selects actions based on the current state, and learns from the rewards received.
2. model.py: The neural network model used by the agent to approximate the Q-values.
3. simulation.py: The environment to interact with the Matlab model which will be used by the agent.
4. run_savedmodel.py: Run the saved model from the previous training.
5. helper.py: To plot the loss of the current training.