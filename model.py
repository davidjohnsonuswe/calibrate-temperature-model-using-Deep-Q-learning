import os
import torch
import torch.nn as nn
import torch.optim as optim


class Linear_QNet(nn.Module):
    """A simple linear neural network for Q-learning.

    This network uses a stack of linear layers with ReLU activations to
    approximate the Q-function in a reinforcement learning context.
    """

    def __init__(self):
        """Initializes the neural network with a predefined architecture.

        The network consists of three linear layers with ReLU activation between
        them. The input layer has 25 features, and the output layer has 17 units,
        corresponding to the expected action space.
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(25, 128),  # Input layer to hidden layer
            nn.ReLU(),  # ReLU activation function
            nn.Linear(128, 128),  # Hidden layer to another hidden layer
            nn.ReLU(),  # ReLU activation function
            nn.Linear(128, 17),  # Output layer
        )

    def forward(self, x):
        """Forward pass through the neural network.

        This method defines how the input data flows through the network to produce
        the output.

        Args:
            x (torch.Tensor): The input tensor with the expected shape for the network.

        Returns:
            torch.Tensor: The output logits from the network.
        """
        logits = self.linear_relu_stack(x)  # Pass input through the network
        return logits

    def save(self, file_name='model.pth'):
        """Saves the model's state to a file.

        This method saves the current state of the model to a specified file.

        Args:
            file_name (str): The name of the file where the model state will be saved.
                             Default is 'model.pth'.
        """
        model_folder_path = './model'  # Directory to save the model
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """A trainer class for training a Q-learning neural network.

    This class handles the training process for a Q-learning model, using a given
    learning rate, discount factor, and loss function.
    """

    def __init__(self, model, lr, gamma):
        """Initializes the trainer with a model, learning rate, and discount factor.

        Args:
            model (nn.Module): The Q-learning model to be trained.
            lr (float): The learning rate for the optimizer.
            gamma (float): The discount factor for future rewards in Q-learning.
        """
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor for future rewards
        self.model = model  # The Q-learning model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # Adam optimizer
        self.criterion = nn.MSELoss()  # Mean Squared Error loss function

    def train_step(self, state, action, reward, next_state, done):
        """Performs a single training step for the Q-learning model.

        This method updates the model based on a single training experience,
        adjusting the predicted Q-values to account for the observed reward and
        estimated future rewards.

        Args:
            state (np.ndarray): The previous state.
            action (np.ndarray): The action taken in that state.
            reward (float): The reward received for the action.
            next_state (np.ndarray): The resulting state after the action.
            done (bool): Whether the simulation has terminated.
        """
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float)  # State tensor
        next_state = torch.tensor(next_state, dtype=torch.float)  # Next state tensor
        action = torch.tensor(action, dtype=torch.long)  # Action tensor
        reward = torch.tensor(reward, dtype=torch.float)  # Reward tensor
        # (n, x)

        # Ensure tensors are in the correct shape
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()  # Zero the gradients before backpropagation
        loss = self.criterion(target, pred)  # Calculate the loss
        loss.backward()  # Backpropagate the gradients

        self.optimizer.step()  # Update the model parameters with the optimizer



