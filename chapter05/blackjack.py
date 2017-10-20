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
    def __init__(self, episodes=500000):
        self.episodes = episodes
        self.poker = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.policy = {}
        self.valueOfState = {}
        self.valueOfStateAction = {}
        self.samples = []

        for psum in range(12,22):
            for seenCard in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for usableAce in ['y', 'n']:
                    self.policy[psum, seenCard, usableAce] = ["sticks"] if psum==20 or psum==21 else ["hits"]
                    self.valueOfState[psum, seenCard, usableAce] = 0.0

    def oneRound(self):
        visitedStates = []
        visitedActions = []
        score = 0.0
        player = Player(np.random.choice(self.poker, size=2, replace=True))
        dealer = Dealer(np.random.choice(self.poker, size=2, replace=True))
        while True:
            if player.isBust():
                score = -1.0
                break
            state = (player.getSum(), dealer.seenCard, player.isUsableAce())
            if state in self.policy.keys():
                action = np.random.choice(self.policy[state])
                visitedStates.append(state)
                visitedActions.append(action)
                if action == "sticks":
                    while(dealer.getSum()<17):
                        dealer.hits(np.random.choice(self.poker))
                    score = 1.0 if dealer.isBust() else self.getWinner(player.getSum(), dealer.getSum())
                    break
            player.hits(np.random.choice(self.poker))
        return (visitedStates, visitedActions, score)

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
                    self.valueOfState[state] = sample[2]
                    stateCounts[state] = 1
                else:
                    self.valueOfState[state] = (1.0*stateCounts[state]*self.valueOfState[state] \
                                                     +sample[2]) / (stateCounts[state]+1)
                    stateCounts[state] += 1

    def estimateStateActionValue(self):
        stateActionCounts = {}
        for sample in self.samples:
            for stateAction in zip(sample[0], sample[1]):
                if not stateActionCounts.has_key(stateAction):
                    self.valueOfStateAction[stateAction] = sample[2]
                    stateActionCounts[stateAction] = 1
                else:
                    self.valueOfStateAction[stateAction] = (1.0*stateActionCounts[stateAction]* \
                                    self.valueOfStateAction[stateAction]+sample[2]) / (stateActionCounts[stateAction]+1)
                    stateActionCounts[stateAction] += 1

    def monteCarloEs(self, episodes=500000):
        self.episodes = episodes
        self.policy = {}
        self.valueOfState = {}
        self.valueOfStateAction = {}
        self.samples = []

        for psum in range(12,22):
            for seenCard in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for usableAce in ['y', 'n']:
                    self.policy[psum, seenCard, usableAce] = ["sticks"] if psum==21 else ["sticks", "hits"]
                    self.valueOfState[psum, seenCard, usableAce] = 0.0
                    self.valueOfStateAction[(psum, seenCard, usableAce), "sticks"] = 0.0
                    self.valueOfStateAction[(psum, seenCard, usableAce), "hits"] = 0.0
        self.simulate()
        self.estimateStateActionValue()
        for state in self.policy.keys():
            if self.valueOfStateAction[(state), "sticks"] > self.valueOfStateAction[(state), "hits"]: 
                self.policy[state] = ["sticks"]
            else:
                self.policy[state] = ["hits"]
        self.simulate()

    def getWinner(self, player, dealer):
        if player > dealer:
            return 1.0
        elif player < dealer:
            return -1.0
        return 0.0

    def printStateValue(self):
        for state in self.valueOfState.keys():
            print("state-value:", state, self.valueOfState[state])

def figure5_1(valueOfState):
    x1, x2 = [], []
    y1, y2 = [], []
    z1, z2 = [], []
    for state in valueOfState.keys():
        if state[2] == 'y':
            x1.append(1 if state[1]==0 else state[1])
            y1.append(state[0])
            z1.append(valueOfState[state])
        else:
            x2.append(1 if state[1]==0 else state[1])
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
    ax.set_zlim(-1,1)
    ax.set_title('Usable ace')

    ax = plt.subplot(212, projection='3d')
    ax.scatter(x2, y2, z2)
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('player sum')
    ax.set_zlabel("estimated value")
    ax.set_xlim(0, 11)
    ax.set_ylim(12, 22)
    ax.set_zlim(-1,1)
    ax.set_title('No usable ace')

    plt.show()

def figure5_3_left(policy):
    x11, x12 = [], []
    y11, y12 = [], []
    x21, x22 = [], []
    y21, y22 = [], []
    for state in policy.keys():
        if state[2] == 'y':
            if policy[state] == ["sticks"]:
                x11.append(1 if state[1]==0 else state[1])
                y11.append(state[0])
            else:
                x12.append(1 if state[1]==0 else state[1])
                y12.append(state[0])
        else:
            if policy[state] == ["sticks"]:
                x21.append(1 if state[1]==0 else state[1])
                y21.append(state[0])
            else:
                x22.append(1 if state[1]==0 else state[1])
                y22.append(state[0])
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(211)
    ax.scatter(x11, y11, label="sticks")
    ax.scatter(x12, y12, label="hits")
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('player sum')
    ax.set_xlim(0, 11)
    ax.set_ylim(11, 22)
    ax.set_xticks(range(0,12))
    ax.legend()
    ax.set_title('Usable ace')

    ax = plt.subplot(212)
    ax.scatter(x21, y21, label="sticks")
    ax.scatter(x22, y22, label="hits")
    ax.set_xlabel('dealer showing')
    ax.set_ylabel('player sum')
    ax.set_xlim(0, 11)
    ax.set_ylim(11, 22)
    ax.set_xticks(range(0,12))
    ax.legend()
    ax.set_title('No usable ace')

    plt.subplots_adjust(hspace=0.5)
    plt.show()

if __name__=='__main__':
    np.random.seed(47)
    blackjack = Blackjack(episodes=500000)
    # fig 5.1
    blackjack.simulate()
    blackjack.estimateStateValue()
    figure5_1(blackjack.valueOfState)

    # fig 5.3 right
    blackjack.monteCarloEs(episodes=3000000)
    blackjack.estimateStateValue()
    figure5_1(blackjack.valueOfState)

    # fig 5.3 left
    figure5_3_left(blackjack.policy)
