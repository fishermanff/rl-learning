"""
@author chenjianfeng
@date 2017.10
"""

from __future__ import print_function
import numpy as np

class BanditGame(object):
    def __init__(self, kArm=10, epsilon=0.0, qMean=0.0, initials=0.0, alpha=0.1, averageSamp=False, ucb=False, 
                            ucbC=2, gradient=False, gradientBaseline=False, maxSteps=1000):
        self.kArm = kArm
        self.epsilon = epsilon
        self.qMean = 0.0
        self.qSigma = np.sqrt(1.0)
        self.rewardSigma = np.sqrt(1.0)
        self.initials = initials
        self.alpha = alpha
        self.averageSamp = averageSamp
        self.ucb = ucb
        self.ucbC = ucbC
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
        self.actionProb = np.zeros(self.kArm)
        self.averReward = 0.0

    def softmax(self, x):
        expVal = np.exp(x)
        return expVal/np.sum(expVal)

    def getAction(self):
        if(np.random.rand()<self.epsilon):
            return self.actions[np.random.randint(self.kArm)]
        if(self.ucb):
            return np.argmax(self.qEst + 
                        np.array([self.ucbC * np.sqrt(np.log(self.times+1)/(Nt+1)) for Nt in self.cumulativeAction]))
        if(self.gradient):
            self.actionProb = self.softmax(self.qEst)
            return np.random.choice(self.actions, p=self.actionProb)
        return np.argmax(self.qEst)

    def doAction(self, action):
        self.times += 1
        reward = np.random.normal(self.qActual[action], self.rewardSigma)
        self.actionAndReward.append((action, reward))
        self.cumulativeAction[action] += 1
        self.cumulativeReward[action] += reward
        if(self.gradient):
            self.averReward = (self.times-1.0)*self.averReward/self.times + reward/self.times
            mask = np.zeros(self.kArm)
            mask[action] = 1
            if(self.gradientBaseline):
                self.qEst = self.qEst + self.alpha*(reward-self.averReward)*(mask-self.actionProb)
            else:
                self.qEst = self.qEst + self.alpha*reward*(mask-self.actionProb)
        elif(self.averageSamp==True):
            self.qEst[action] = self.qEst[action] + (reward-self.qEst[action])/self.cumulativeAction[action]
        else:
            self.qEst[action] = self.qEst[action] + self.alpha*(reward-self.qEst[action])


    def run(self):
        for step in range(self.maxSteps):
            action = self.getAction()
            self.doAction(action)


