import gym
import tree_env
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torch.nn.functional as F
import bcolz


env = gym.make('tree-v0').unwrapped
vocab_file = 'vocab.pkl'
train_file = 'sick_train_deptree.txt'
test_file = 'sick_test_deptree.txt'
glove_path = '../data/glove.6B'
save_path = '../models/save.pth'
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


EMBEDDING_DIM = 50
HIDDEN_DIM = 50

##################
episode_durations = []

### Get vocabulary ###
with open(vocab_file, 'rb') as vocab:
    vocab_dict = pickle.load(vocab)

vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
glove = {w: vectors[word2idx[w]] for w in words}


# build vocab matrix
matrix_len = len(vocab_dict)
weights_matrix = np.zeros((matrix_len, EMBEDDING_DIM))
words_found = 0

# assign glove vector if found, else assign random vector
for i, word in enumerate(vocab_dict.keys()):
    try:
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM, ))
        word2idx[word] = len(word2idx)

def dep_tree_to_sent(sentence_tree): # tested
    words = [[int(node[2][2]), node[2][0]] for node in sentence_tree]
    words = sorted(words)
    sent_list = [tup[1] for tup in words]
    return sent_list

def prepare_sequence(premise, hypothesis):
    prem_list = dep_tree_to_sent(premise)
    hypo_list = dep_tree_to_sent(hypothesis)
    sentence = hypo_list + prem_list
    indices = [vocab_dict[w] for w in sentence]
    return torch.tensor(indices, dtype=torch.long, device=device)


### DQN Module ###
class DQN(nn.Module):
    # input is concatenated hypothesis, premise tree pair.
    # ['This', 'is', 'the', 'hypothesis'] + ['This', 'is', 'the', 'premise']
    # outputs are q values for 5 * MAX_SENTENCE_SIZE * MAX_SENTENCE_SIZE actions.
    def __init__(self, embedding_dim, hidden_dim, outputs):
        super(DQN, self).__init__()
        self.hidden_dim = hidden_dim
        num_embeddings, temp = weights_matrix.shape
        self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(weights_matrix))
        # self.word_embeddings.load_state_dict({'weight':weights_matrix})
        # also learn embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.nn1 = nn.Linear(hidden_dim, hidden_dim)
        self.nn2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, outputs)

    def forward(self, sentence_idx):
        embeds = self.word_embeddings(sentence_idx)
        _, (h_n, _) = self.lstm(embeds.view(len(sentence_idx), 1, -1))
        # h_n => last cell's hidden state
        # print("LSTM_out", h_n.view(h_n.size(0), -1).shape)
        x = self.nn1(h_n.view(h_n.size(0), -1))
        x = self.nn2(x.view(x.size(0), -1))
        return self.head(x.view(x.size(0), -1))


### Training loop ###
num_episodes = 50
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20
TARGET_UPDATE = 10
NUM_EPOCHS = 20

MAX_SENTENCE_SIZE = 20
# n_actions = env.action_space.n
n_actions = 3
VOCAB_SIZE = len(vocab_dict)
print("Num_actions = ", n_actions)
policy_net = DQN(EMBEDDING_DIM, HIDDEN_DIM, n_actions).to(device)
target_net = DQN(EMBEDDING_DIM, HIDDEN_DIM, n_actions).to(device)
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
            temp = policy_net(state)
            return temp.max(1)[1].view(1, 1)
            # return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def best_decision(state):
    with torch.no_grad():
        max_act = -1
        max_val = -10000
        profits = policy_net(state)
        for action in range(3):
            if profits[action] > max_val:
                max_act = action
                max_val = profits[action]
        return max_act

### Updates from replay memory ###
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # Compute Q value from policy net.
    state_action_values = torch.zeros(BATCH_SIZE, 1, device=device)
    for i in range(BATCH_SIZE):
        state_action_values[i] = policy_net(batch.state[i])[0][batch.action[i]]
    reward_batch = torch.cat(batch.reward)
    # Get expected max value for the state from target net.
    next_state_values = torch.zeros(BATCH_SIZE, 1, device=device)

    for i in range(BATCH_SIZE):
        if batch.next_state[i] is not None:
            next_state_values[i] = target_net(batch.next_state[i]).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def read_sentences(line):
    line = line.split('\n')[0]
    a = line.split('\t')
    label = a[2].strip()
    sent1_space = a[0].strip().split(' ')
    sent2_space = a[1].strip().split(' ')
    sent1 = []
    sent2 = []
    for i in range(len(sent1_space)):
        node_raw = sent1_space[i].split(',')
        par_id = node_raw[2]
        if '.' in par_id:
            par_id = float(par_id)
        else:
            par_id = int(par_id)
        child_id = int(node_raw[6])
        node = [[node_raw[0], node_raw[1], par_id], node_raw[3], [node_raw[4], node_raw[5], child_id]]
        sent1.append(node)
    for i in range(len(sent2_space)):
        node_raw = sent2_space[i].split(',')
        par_id = node_raw[2]
        if '.' in par_id:
            par_id = float(par_id)
        else:
            par_id = int(par_id)
        child_id = int(node_raw[6])
        node = [[node_raw[0], node_raw[1], par_id], node_raw[3], [node_raw[4], node_raw[5], child_id]]
        sent2.append(node)
    return sent1, sent2, label

### Learn from each training example ###
with open('../data/' + train_file, 'r') as training_data:
    for line_num, line in enumerate(training_data):
        if(len(line) == 1):
            continue
        sent1, sent2, label = read_sentences(line)
        # Run MAX_EPISODES episodes on each training example.
        for episode in range(num_episodes):
            env.reset()
            env.setParams(sent1, sent2, label)
            state = prepare_sequence(sent1, sent2)
            for t in count():
                action = select_action(state)
                _, reward, done, _ = env.step(action.item())

                reward = torch.tensor([[reward]], device=device)
                print(t, reward)
                next_state = prepare_sequence(env.premise_tree, env.hypothesis_tree)
                if done:
                    next_state = None
                # print(state, env.decode_action(action.item()), reward)
                # Store transition into memory
                memory.push(state, action, next_state, reward)
                state = next_state
                optimize_model()
                if done:
                    episode_durations.append(t+1)
                    break
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

# save model
torch.save(policy_net.state_dict(), save_path)

# load model
# model = DQN(EMBEDDING_DIM, HIDDEN_DIM, n_actions).to(device)
# model.load_state_dict(torch.load(save_path))
# model.eval()
# Test now.
correct = 0
total = 0
policy_net.eval() # In evaluation mode now.
with open('../data/' + test_file, 'r') as test_data:
    for line in test_data:
        sent1, sent2, label = read_sentences(line)
        state = prepare_sequence(sent1, sent2)
        action = best_decision(state)
        gold = -1
        if label == 'ENTAILMENT':
            gold = 0
        elif label == 'NEUTRAL':
            gold = 1
        else:
            gold = 2
        if action == gold:
            correct += 1
        total += 1

print("Accuracy: ", correct,"/", total)