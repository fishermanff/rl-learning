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
    def __init__(self, maxSteps=1000, runCnt=2000, epsilon=0.0, qMean=0.0, initials=0.0,
                    alpha=0.1, averageSamp=False, ucb=False, ucbC=2, gradient=False, gradientBaseline=False):
        self.maxSteps = maxSteps
        self.runCnt = runCnt
        self.epsilon = epsilon
        self.qMean = qMean
        self.initials = initials
        self.alpha = alpha
        self.averageSamp = averageSamp
        self.ucb = ucb
        self.ucbC = ucbC
        self.gradient = gradient
        self.gradientBaseline = gradientBaseline
        self.action = []
        self.reward = []
        self.isCorrectAction = []

    def run(self):
        bandits = [BanditGame(maxSteps=self.maxSteps, epsilon=self.epsilon, qMean=self.qMean, initials=self.initials,
                                alpha=self.alpha, averageSamp=self.averageSamp, ucb=self.ucb, ucbC=self.ucbC, 
                                gradient=self.gradient, gradientBaseline=self.gradientBaseline) for _ in range(self.runCnt)]
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

    def averRewardOfSteps(self):
        return np.array(self.reward)[:,self.maxSteps-1].mean()


def figure2_1():
    banditGame = BanditGame()
    plt.figure("fig2.1")
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

    plt.figure("fig2.2")
    plt.subplot(211)
    for (idx,color,name) in zip(range(len(averageReward)), ['r','g','b'], ["epsilon=0.0", "epsilon=0.01", "epsilon=0.1"]):
        plt.plot(np.arange(maxSteps), averageReward[idx], color, label=name)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average reward")

    plt.subplot(212)
    for (idx,color,name) in zip(range(len(correctPercent)), ['r','g','b'], ["epsilon=0.0", "epsilon=0.01", "epsilon=0.1"]):
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

    plt.figure("fig2.3")
    for (ex, color, name) in zip([ex1, ex2], ['r','b'], ["$epsilon=0.0, Q_1=5.0$", "$epsilon=0.1, Q_1=0.0$"]):
        plt.plot(np.arange(maxSteps), ex.getCorrectPercent(), color, label=name)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Optimal action")
    plt.show()

def figure2_4():
    maxSteps = 1000
    ex1 = BanditRun(maxSteps=maxSteps, ucb=True)
    ex2 = BanditRun(maxSteps=maxSteps, epsilon=0.1)
    ex1.run()
    ex2.run()

    plt.figure("fig2.4")
    for (ex, color, name) in zip([ex1, ex2], ['r','b'], ["$UCB, c=2$", r"$\varepsilon-greedy, epsilon=0.1$"]):
        plt.plot(np.arange(maxSteps), ex.getAverageReward(), color, label=name)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()

# fig2.5 gradientBaseline=False的两条曲线与原文不一致，原因未查明
def figure2_5():
    maxSteps = 1000
    ex1 = BanditRun(alpha=0.1, gradient=True, gradientBaseline=True, qMean=4.0)
    ex2 = BanditRun(alpha=0.4, gradient=True, gradientBaseline=True, qMean=4.0)
    ex3 = BanditRun(alpha=0.1, gradient=True, gradientBaseline=False, qMean=4.0)
    ex4 = BanditRun(alpha=0.4, gradient=True, gradientBaseline=False, qMean=4.0)
    ex1.run()
    ex2.run()
    ex3.run()
    ex4.run()

    plt.figure("fig2.5")
    names = [r"$\alpha=0.1, with baseline$",
             r"$\alpha=0.4, with baseline$",
             r"$\alpha=0.1, without baseline$",
             r"$\alpha=0.4, without baseline$"]
    for (ex, name) in zip([ex1, ex2, ex3, ex4], names):
        plt.plot(np.arange(maxSteps), ex.getCorrectPercent(), label=name)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Optimal action")
    plt.show()

def figure2_6():
    maxSteps = 1000
    parameters = {'epsilon': [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0], 
                  'alpha': [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0], 
                  'ucbC': [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0], 
                  'qInit': [-2.0, -1.0, 0.0, 1.0, 2.0]}

    plt.figure("fig2.6")
    ex1 = [BanditRun(epsilon=pow(2,x)) for x in parameters['epsilon']]
    ex2 = [BanditRun(gradient=True, gradientBaseline=True, alpha=pow(2,x)) for x in parameters['alpha']]
    ex3 = [BanditRun(ucb=True, alpha=0.1, ucbC=pow(2,x)) for x in parameters['ucbC']]
    ex4 = [BanditRun(epsilon=0.0, initials=pow(2,x)) for x in parameters['qInit']]
    for exGroup in [ex1, ex2, ex3, ex4]:
        for ex in exGroup:
            ex.run()
    plt.plot(parameters['epsilon'], [ex.averRewardOfSteps() for ex in ex1], label=r"$\varepsilon-greedy$")
    plt.plot(parameters['alpha'], [ex.averRewardOfSteps() for ex in ex2], label="gradient-bandit")
    plt.plot(parameters['ucbC'], [ex.averRewardOfSteps() for ex in ex3], label="UCB")
    plt.plot(parameters['qInit'], [ex.averRewardOfSteps() for ex in ex4], label=r"$q-Initials(\alpha=0.1)$")
    plt.legend()
    plt.xlabel("para.=2^x")
    plt.ylabel("Average reward")
    plt.show()

if __name__ == '__main__':
    np.random.seed(47)
    figure2_1()
    figure2_2()
    figure2_3()
    figure2_4()
    figure2_5()
    figure2_6()

