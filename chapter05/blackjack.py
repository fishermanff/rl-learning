"""
@author chenjianfeng
@date 2017.10
"""

from __future__ import print_function
from dealer import Dealer
from player import Player
import numpy as np
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D

class Blackjack(object):
    def __init__(self, episodes=100):
        self.episodes = episodes
        self.poker = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.policy = {}
        self.valueOfState = {}

        self.samples = []

        for psum in range(12,22):
            for seenCard in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for usableAce in ['y', 'n']:
                    self.policy[psum, seenCard, usableAce] = "sticks" if psum==20 or psum==21 else "hits"
                    self.valueOfState[psum, seenCard, usableAce] = 0.0

    def oneRound(self):
        visitedStates = []
        score = 0.0
        player = Player(np.random.choice(self.poker, size=2, replace=True))
        dealer = Dealer(np.random.choice(self.poker, size=2, replace=True))
        while True:
            if player.isBust():
                score = -1.0
                break
            state = (player.getSum(), dealer.seenCard, player.isUsableAce())
            if state in self.policy.keys():
                visitedStates.append(state)
                if self.policy[state] == "sticks":
                    while(dealer.getSum()<17):
                        dealer.hits(np.random.choice(self.poker, size=1, replace=True)[0])
                    score = 1.0 if dealer.isBust() else self.getWinner(player.getSum(), dealer.getSum())
                    break
            player.hits(np.random.choice(self.poker, size=1, replace=True)[0])
        return (visitedStates, score)

    def simulate(self):
        episodes = 0
        while episodes<self.episodes:
            if episodes%1000 == 0:
                print("episode %d simulating ..." %(episodes))
            self.samples.append(self.oneRound())
            episodes += 1

    def estimateStateValue(self):
        stateCounts = {}
        for sample in self.samples:
            for state in sample[0]:
                if not stateCounts.has_key(state):
                    self.valueOfState[state] = sample[1]
                    stateCounts[state] = 1
                else:
                    self.valueOfState[state] = (1.0*stateCounts[state]*self.valueOfState[state] \
                                                     +sample[1]) / (stateCounts[state]+1)
                    stateCounts[state] += 1 

    def getWinner(self, player, dealer):
        if player > dealer:
            return 1.0
        elif player < dealer:
            return -1.0
        return 0.0

    def printSamples(self):
        for sample in self.samples:
            print(sample)

    def printStateValue(self):
        for state in self.valueOfState.keys():
            print("state-value:", state, self.valueOfState[state])

def figure5_1(valueOfState):
    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    for state in valueOfState.keys():
        if state[2] == 'y':
            x1.append(state[1])
            y1.append(state[0])
            z1.append(valueOfState[state])
        else:
            x2.append(state[1])
            y2.append(state[0])
            z2.append(valueOfState[state])

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(211, projection='3d')
    ax.scatter(x1, y1, z1)
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('player sum')
    ax.set_zlabel("estimated value")
    ax.set_xlim(0, 11)
    ax.set_ylim(12, 22)
    ax.set_title('Usable ace')

    ax = plt.subplot(212, projection='3d')
    ax.scatter(x2, y2, z2)
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('player sum')
    ax.set_zlabel("estimated value")
    ax.set_xlim(0, 11)
    ax.set_ylim(12, 22)
    ax.set_title('No usable ace')

    plt.show()

if __name__=='__main__':
    np.random.seed(47)
    blackjack = Blackjack(episodes=500000)
    blackjack.simulate()
    blackjack.estimateStateValue()
    figure5_1(blackjack.valueOfState)
