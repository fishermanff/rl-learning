"""
@author chenjianfeng
@date 2017.10
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bandit import BanditGame

class BanditRun(object):
    def __init__(self, maxSteps=1000, runCnt=2000, epsilon=0.1, initials=0.0, averageSamp=False):
        self.maxSteps = maxSteps
        self.runCnt = runCnt
        self.epsilon = epsilon
        self.initials = initials
        self.averageSamp = averageSamp
        self.action = []
        self.reward = []
        self.isCorrectAction = []

    def run(self):
        bandits = [BanditGame(maxSteps=self.maxSteps, epsilon=self.epsilon, initials=self.initials, \
                                averageSamp=self.averageSamp) for _ in range(self.runCnt)]
        for bd in bandits:
            bd.run()
            actionList = [x[0] for x in bd.actionAndReward]
            self.action.append(actionList)
            self.reward.append([x[1] for x in bd.actionAndReward])
            self.isCorrectAction.append([1 if actionList[i]==bd.bestAction else 0 for i in range(len(actionList))])

    def getAverageReward(self):
        return np.array(self.reward).mean(axis=0)

    def getCorrectPercent(self):
        return np.array(self.isCorrectAction).mean(axis=0)


def figure2_1():
    banditGame = BanditGame()
    plt.figure("Figure 2.1")
    sns.violinplot(data=np.random.randn(10000, banditGame.kArm) + banditGame.qActual)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.show()

def figure2_2():
    maxSteps = 1000
    eps = [0.0, 0.01, 0.1]
    averageReward = []
    correctPercent = []
    for ep in eps:
        banditRun = BanditRun(maxSteps=maxSteps, epsilon=ep, averageSamp=True)
        banditRun.run()
        averageReward.append(banditRun.getAverageReward())
        correctPercent.append(banditRun.getCorrectPercent())

    plt.figure("Figure 2.2")
    plt.subplot(211)
    for (idx,color,name) in zip(range(len(averageReward)), ['r','g','b'], ["epsilon=0.0", "epsilon=0.01", "epsilon=0.1"]):
        plt.plot(np.arange(maxSteps), averageReward[idx], color, label=name)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average reward")

    plt.subplot(212)
    for (idx,color,name) in (zip(range(len(correctPercent)), ['r','g','b'], ["epsilon=0.0", "epsilon=0.01", "epsilon=0.1"])):
        plt.plot(np.arange(maxSteps), correctPercent[idx], color, label=name)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Optimal action")
    plt.show()

def figure2_3():
    maxSteps = 1000
    ex1 = BanditRun(maxSteps=maxSteps, epsilon=0.0, initials=5.0)
    ex2 = BanditRun(maxSteps=maxSteps, epsilon=0.1, initials=0.0)
    ex1.run()
    ex2.run()

    plt.figure("Figure 2.3")
    for (ex, color, name) in zip([ex1, ex2], ['r','b'], ["$epsilon=0.0, Q_1=5.0$", "$epsilon=0.1, Q_1=0.0$"]):
        plt.plot(np.arange(maxSteps), ex.getCorrectPercent(), color, label=name)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Optimal action")
    plt.show()

if __name__ == '__main__':
    np.random.seed(47)
    figure2_1()
    figure2_2()
    figure2_3()

