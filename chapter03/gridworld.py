"""
@author chenjianfeng
@date 2017.10
"""

import numpy as np

class GridWorld(object):
    def __init__(self):
        self.gridWidth = 5
        self.gridHeight = 5
        self.grid = np.zeros((self.gridWidth, self.gridHeight))
        self.actions = ['E', 'S', 'W', 'N']
        self.posOfA = (0, 1)
        self.posOfB = (0, 3)
        self.posOfAp = (4, 1)
        self.posOfBp = (2, 3)

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
                reward = {'E':0.0, 'S':0.0, 'W':0.0, 'N':0.0}
                if(i==0):
                    reward['N'] = -1.0
                if(i==self.gridHeight-1):
                    reward['S'] = -1.0
                if(j==0):
                    reward['W'] = -1.0
                if(j==self.gridWidth-1):
                    reward['E'] = -1.0
                if((i,j) == self.posOfA):
                    for key in reward:
                        reward[key] = 10.0
                if((i,j) == self.posOfB):
                    for key in reward:
                        reward[key] = 5.0
                self.rewardOfStateAction[i,j] = reward

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
                if((i,j) == self.posOfA):
                    for key in successor:
                        successor[key] = self.posOfAp
                if((i,j) == self.posOfB):
                    for key in successor:
                        successor[key] = self.posOfBp
                self.successorOfStateAction[i,j] = successor

    def calculateRandomGrid(self, epoch=1000, discount=0.9):
        while epoch>0:
            newGrid = np.zeros((5,5))
            for i in range(self.gridWidth):
                for j in range(self.gridHeight):
                    for action in self.actions:
                        newGrid[i,j] += self.actionProbOfState[i,j][action] * (self.rewardOfStateAction[i,j][action] + \
                                                discount * self.grid[self.successorOfStateAction[i,j][action]])
            self.grid = newGrid
            epoch = epoch - 1

if __name__ == '__main__':
    gridWorld = GridWorld()
    gridWorld.calculateRandomGrid()
    print(gridWorld.grid)
