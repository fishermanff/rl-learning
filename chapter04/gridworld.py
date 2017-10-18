"""
@author chenjianfeng
@date 2017.10
"""

from __future__ import print_function
import numpy as np

class GridWorld(object):
    def __init__(self):
        self.gridHeight = 4
        self.gridWidth = 4
        self.grid = np.zeros((self.gridHeight, self.gridWidth))
        self.actions = ['E', 'S', 'W', 'N']
        self.terminals = [(0,0), (3,3)]

        self.actionProbOfState = {}
        self.rewardOfStateAction = {}
        self.successorOfStateAction = {}
        self.setActionProbOfState()
        self.setRewardOfStateAction()
        self.setSuccessorOfStateAction()

    def setActionProbOfState(self):
        for i in range(self.gridHeight):
            for j in range(self.gridWidth):
                self.actionProbOfState[i,j] = {'E':0.25, 'S':0.25, 'W':0.25, 'N':0.25}

    def setRewardOfStateAction(self):
        for i in range(self.gridHeight):
            for j in range(self.gridWidth):
                self.rewardOfStateAction[i,j] = {'E':-1.0, 'S':-1.0, 'W':-1.0, 'N':-1.0}

    def setSuccessorOfStateAction(self):
        for i in range(self.gridHeight):
            for j in range(self.gridWidth):
                successor = {'E':(i,j+1), 'S':(i+1,j), 'W':(i,j-1), 'N':(i-1,j)}
                if(i==0):
                    successor['N'] = (i,j)
                if(i==self.gridHeight-1):
                    successor['S'] = (i,j)
                if(j==0):
                    successor['W'] = (i,j)
                if(j==self.gridWidth-1):
                    successor['E'] = (i,j)
                self.successorOfStateAction[i,j] = successor

    def calculateRandomGrid(self, epoch=1000, discount=1.0):
        k = 0
        while k<epoch:
            if(k in [0, 1, 2, 3, 10]):
                print("k =", k)
                print(self.grid.round(1))
            newGrid = np.zeros((self.gridHeight, self.gridWidth))
            for i in range(self.gridHeight):
                for j in range(self.gridWidth):
                    if((i,j) not in self.terminals):
                        for action in self.actions:
                            newGrid[i,j] += self.actionProbOfState[i,j][action] * (self.rewardOfStateAction[i,j][action] + \
                                                    discount * self.grid[self.successorOfStateAction[i,j][action]])
            if(np.sum(np.abs(self.grid - newGrid)) < 1e-4):
                self.grid = newGrid
                break
            self.grid = newGrid
            k += 1
        self.grid = self.grid.round(1)

if __name__ == '__main__':
    gridWorld = GridWorld()
    gridWorld.calculateRandomGrid()
    print("random policy grid: ")
    print(gridWorld.grid)


