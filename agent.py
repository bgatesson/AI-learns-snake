import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.nb_games = 0
        self.epsilon = 0 # randomness param
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # calls popleft() if max capacity reached
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snakebody[0]
        head_l = (head[0] - 25, head[1])
        head_r = (head[0] + 25, head[1])
        head_d = (head[0], head[1] + 25)
        head_u = (head[0], head[1] - 25)

        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_d = game.direction == 'DOWN'
        dir_u = game.direction == 'UP'

        state = [
            # danger straight
            (dir_l and game.is_collision(head_l)) or
            (dir_r and game.is_collision(head_r)) or
            (dir_d and game.is_collision(head_d)) or
            (dir_u and game.is_collision(head_u)),

            # danger right
            (dir_d and game.is_collision(head_l)) or
            (dir_u and game.is_collision(head_r)) or
            (dir_r and game.is_collision(head_d)) or
            (dir_l and game.is_collision(head_u)),

            # danger left
            (dir_u and game.is_collision(head_l)) or
            (dir_d and game.is_collision(head_r)) or
            (dir_l and game.is_collision(head_d)) or
            (dir_r and game.is_collision(head_u)),

            # move direction
            dir_l,
            dir_r,
            dir_d,
            dir_u,

            # food location
            game.food[0] < game.snakehead[0], # food is left 
            game.food[0] > game.snakehead[0], # food is right 
            game.food[1] > game.snakehead[1], # food is down 
            game.food[1] < game.snakehead[1], # food is up 
        ]
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # if MAX_MEMORY reached --> popleft

    def train_LM(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*mini_sample) 
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_SM(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves --> explore vs. exploit
        self.epsilon = 80 - self.nb_games
        move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
            move[idx] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            idx = torch.argmax(prediction).item()
            move[idx] = 1

        return move

def train():
    scores = []
    mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get current state
        old_state = agent.get_state(game)

        # get move 
        move = agent.get_action(old_state)

        # perform move and get new state
        reward, game_over, score = game.play(move)
        new_state = agent.get_state(game)

        # train short memory (SM)
        agent.train_SM(old_state, move, reward, new_state, game_over)

        # remember
        agent.remember(old_state, move, reward, new_state, game_over)

        if game_over:
            # train long memory (LM) / replay
            game.reset()
            agent.nb_games += 1
            agent.train_LM()

            if score > best_score:
                best_score = score
                agent.model.save()

            print('Game', agent.nb_games, 'Score', score, 'Best:', best_score)

            scores.append(score)
            total_score += score
            mean = total_score / agent.nb_games
            mean_scores.append(mean)
            plot(scores, mean_scores)

if __name__ == '__main__':
    train()