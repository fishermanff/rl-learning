"""
@author chenjianfeng
@date 2017.10
"""

from __future__ import print_function

class Dealer(object):
    def __init__(self, firstTwoCards):
        self.seenCard = firstTwoCards[0]
        self.blindCard = firstTwoCards[1]
        self.cards = [self.seenCard, self.blindCard]

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

    def printCards(self):
        print("dealer's cards:", self.cards)

