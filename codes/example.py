import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


env = gym.make('CartPole-v0').unwrapped

print("Environment Loaded.")
#gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Replay Memory ###
Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


### DQN Module ###
class DQN():
    def __init__(self):
        pass
    pass

##################
episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


### Training loop ###
num_episodes = 50
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20
TARGET_UPDATE = 10

MAX_SENTENCE_SIZE = 20
env.newInit()
n_actions = env.action_space.n


policy_net = DQN(treelstm_inputs, n_actions).to(device)
target_net = DQN(treelstm_inputs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()   # Do not train target network

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


### Exploration - exploitation step. ###
steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done/ EPS_DECAY)
    steps_done += 1
    # pick action with the largest reward
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

### Updates from replay memory ###
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute current Q from policy net.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Get target Q values from target net.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


### Learn from each training example ###
for ex in range(len(training_data)):
    sent1 = training_data[ex][0]
    sent2 = training_data[ex][1]
    label = training_data[ex][2]

    # Run MAX_EPISODES episodes on each training example.
    env.setParams(sent1, sent2, label)
    for episode in range(num_episodes):
        env.reset()
        state = get_treelstm_rep_tree(sent1)
        for t in count():
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())

            reward = torch.tensor([reward], device=device)

            # TODO: Observe new state here
            next_state = get_treelstm_rep_tree(env.get_sentence())
            if not done:
                next_state = state
            else:
                next_state = None

            # Store transition into memory
            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()
            if done:
                episode_durations.append(t+1)
                plot_durations()    #for debug
                break
