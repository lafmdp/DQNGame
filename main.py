from ATGame import Game
from DQNAlgo import DQN
import random

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MOVE_LEFT = [1, 0, 0]
MOVE_STAY = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

ACTION_SPACE = [MOVE_LEFT, MOVE_STAY, MOVE_RIGHT]

N_START_TRAINING = 1024
TRAIN_EPOCH = 100
SAVE_EPOCH = 20000

def main():
    Agent = DQN(len(ACTION_SPACE))
    game = Game()

    action = random.choice(ACTION_SPACE)
    reward, image = game.step(action)

    n = 0

    while True:
        action = Agent.choose_action(image)
        reward, image2 = game.step(action)

        Agent.save_replay(image, action, reward, image2)

        # the interation step
        image = image2
        n += 1

        print('step: ',n,' reward: ',reward)

        if n > N_START_TRAINING:
            if n % TRAIN_EPOCH is 0:
                Agent.train_network()

            if n % SAVE_EPOCH is 0:
                Agent.save_model()

if __name__ == '__main__':
    main()