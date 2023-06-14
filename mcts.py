from __future__ import division, annotations

from typing import Callable, Any

import time

import numpy as np


def randomPolicy(state: Any, rng: np.random.Generator = np.random.default_rng()) -> float:
    while not state.isTerminal():
        try:
            action = rng.choice(state.getPossibleActions())
        except ValueError:
            raise Exception(f'Non-terminal state has no possible actions: {state}')
        state = state.takeAction(action)
    return state.getReward()


class TreeNode:

    isTerminal: bool
    isFullExpanded: bool
    parent: TreeNode
    numVisits: int
    totalReward: float
    children: dict

    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        return f'{self.__class__.__name__} - totalReward: {self.totalReward}, numVisits: {self.numVisits:d}, ' \
               f'isTerminal: {self.isTerminal}, possibleActions: {self.children.keys()}'


class MCTS:

    rng: np.random.Generator
    timeLimit: float | int
    searchLimit: int
    explorationConstant: float
    rollout: Callable[[Any, np.random.Generator], float]

    def __init__(
        self,
        timeLimit: float | int = None,
        iterationLimit: int = None,
        explorationConstant: float = 1 / np.sqrt(2),
        rolloutPolicy: Callable[[Any, np.random.Generator], float] = randomPolicy,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.root = None

        self.rng = rng

        if timeLimit is not None:
            if iterationLimit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState, returnBestAction=False, needDetails=False):
        self.root = TreeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        if returnBestAction:
            bestChild = self.getBestChild(self.root, 0)
            action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
            if needDetails:
                return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node.state, self.rng)
        self.backpropogate(node, reward)

    def selectNode(self, node: TreeNode) -> TreeNode:
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    @staticmethod
    def expand(node: TreeNode) -> TreeNode:
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = TreeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode
        raise Exception("Should never reach here")

    @staticmethod
    def backpropogate(node: TreeNode, reward: float | int):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node: TreeNode, explorationValue: float | int) -> TreeNode:
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * \
                np.sqrt(2 * np.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return self.rng.choice(bestNodes)

    def getBestRoute(self):
        nodeList = []
        treeNode = self.root
        while not treeNode.isTerminal:
            nodeList.append(treeNode.state)
            treeNode = self.getBestChild(treeNode, 0)
        return nodeList
