"""
@author chenjianfeng
@date 2017.10
"""

from __future__ import print_function
import numpy as np

class BanditGame(object):
    def __init__(self, kArm=10, epsilon=0.0, qMean=0.0, qVariance=1.0, 
                    rewardVariance=1.0, initials=0.0, alpha=0.1, averageSamp=False, maxSteps=1000):
        self.kArm = kArm
        self.epsilon = epsilon
        self.qMean = qMean
        self.qSigma = np.sqrt(qVariance)
        self.rewardSigma = np.sqrt(rewardVariance)
        self.initials = initials
        self.alpha = alpha
        self.averageSamp = averageSamp
        self.maxSteps = maxSteps

        self.actions = np.arange(self.kArm)
        self.qActual = np.random.normal(0.0, 1.0, self.kArm)
        self.qEst = [self.initials] * self.kArm
        self.actionAndReward = []

        self.cumulativeAction = {}
        self.cumulativeReward = {}
        self.bestAction = np.argmax(self.qActual)

        for action in self.actions:
            self.cumulativeAction[action] = 0
            self.cumulativeReward[action] = 0

    def getAction(self):
        if(np.random.rand()<self.epsilon):
            return self.actions[np.random.randint(self.kArm)]
        return np.argmax(self.qEst)

    def doAction(self, action):
        reward = np.random.normal(self.qActual[action], self.rewardSigma)
        self.actionAndReward.append((action, reward))
        self.cumulativeAction[action] += 1
        self.cumulativeReward[action] += reward
        if(self.averageSamp==True):
            self.qEst[action] = self.qEst[action] + (reward-self.qEst[action])/self.cumulativeAction[action]
        else:
            self.qEst[action] = self.qEst[action] + self.alpha*(reward-self.qEst[action])

    def run(self):
        for step in range(self.maxSteps):
            action = self.getAction()
            self.doAction(action)


