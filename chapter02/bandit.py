"""
@author chenjianfeng
@date 2017.10
"""

from __future__ import print_function
import numpy as np

class BanditGame(object):
    def __init__(self, kArm=10, epsilon=0.0, qMean=0.0, initials=0.0, alpha=0.1, averageSamp=False, ucb=False, 
                            gradient=False, gradientBaseline=False, maxSteps=1000):
        self.kArm = kArm
        self.epsilon = epsilon
        self.qMean = 0.0
        self.qSigma = np.sqrt(1.0)
        self.rewardSigma = np.sqrt(1.0)
        self.initials = initials
        self.alpha = alpha
        self.averageSamp = averageSamp
        self.ucb = ucb
        self.gradient = gradient
        self.gradientBaseline = gradientBaseline
        self.maxSteps = maxSteps

        self.actions = np.arange(self.kArm)
        self.qActual = np.random.normal(self.qMean, self.qSigma, self.kArm)
        self.qEst = np.array([self.initials] * self.kArm)
        self.actionAndReward = []

        self.cumulativeAction = np.zeros(self.kArm)
        self.cumulativeReward = np.zeros(self.kArm)
        self.bestAction = np.argmax(self.qActual)
        self.times = 0
        self.confidence = 2
        self.preference = np.zeros(self.kArm)
        self.probability = np.zeros(self.kArm)

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))

    def getAction(self):
        if(self.ucb):
            return np.argmax(self.qEst + 
                        np.array([self.confidence * np.sqrt(np.log(self.times+1)/(Nt+1)) for Nt in self.cumulativeAction]))
        if(self.gradient):
            self.probability = self.softmax(self.preference)
            return np.argmax(self.probability)
        if(np.random.rand()<self.epsilon):
            return self.actions[np.random.randint(self.kArm)]
        return np.argmax(self.qEst)

    def doAction(self, action):
        self.times += 1
        reward = np.random.normal(self.qActual[action], self.rewardSigma)
        self.actionAndReward.append((action, reward))
        self.cumulativeAction[action] += 1
        self.cumulativeReward[action] += reward
        if(self.gradient):
            mask = np.array([1 if x==action else 0 for x in self.actions])
            if(self.gradientBaseline):
                self.preference = self.preference + \
                                    self.alpha*(reward-1.0*self.cumulativeReward[action]/self.times)*(mask-self.probability)
            else:
                self.preference = self.preference + self.alpha*reward*(mask-self.probability)
        else:
            if(self.averageSamp==True):
                self.qEst[action] = self.qEst[action] + (reward-self.qEst[action])/self.cumulativeAction[action]
            else:
                self.qEst[action] = self.qEst[action] + self.alpha*(reward-self.qEst[action])


    def run(self):
        for step in range(self.maxSteps):
            action = self.getAction()
            self.doAction(action)


