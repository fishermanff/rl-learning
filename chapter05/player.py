"""
@author chenjianfeng
@date 2017.10
"""

from __future__ import print_function

class Player(object):
    def __init__(self, firstTwoCards):
        self.cards = [firstTwoCards[0], firstTwoCards[1]]

    def getSum(self):
        if 0 not in self.cards:
            return sum(self.cards)
        if(sum(self.cards)<11):
            return sum(self.cards) + 11 + self.cards.count(0) - 1
        else:
            return sum(self.cards) + self.cards.count(0)

    def hits(self, card):
        self.cards.append(card)

    def isBlackjack(self):
        return len(self.cards)==2 and self.getSum()

    def isBust(self):
        return self.getSum()>21

    def isUsableAce(self):
        if 0 not in self.cards:
            return 'n'
        return 'y' if sum(self.cards)<11 else 'n'

    def printCards(self):
        print("player's cards:", self.cards)