import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 1. The Neural Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Simple 3-layer network
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim) # Output: [Q_Red, Q_Green]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. The Agent (The Brain)
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99          # Discount factor (Future rewards are slightly less important)
        self.epsilon = 1.0         # Exploration rate (Start by acting randomly)
        self.epsilon_min = 0.01    # Minimum exploration
        self.epsilon_decay = 0.95 # How fast to stop exploring
        self.learning_rate = 0.001
        self.batch_size = 32

        # The Brains
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Memory (Experience Replay)
        self.memory = deque(maxlen=2000)

    def act(self, state):
        # Exploration: Random Move
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: Best Move
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # Train the model using a random batch from memory
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)

            # The Target: Reward + Best Guess for Future
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()

            # The Prediction: What the model currently thinks
            target_f = self.model(state_tensor)
            current_prediction = target_f.clone()
            current_prediction[action] = target

            # Update weights
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, current_prediction)
            loss.backward()
            self.optimizer.step()

    def decay_epsilon(self):
        # Create a specific function for this
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay