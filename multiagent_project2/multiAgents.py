# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Evaluate the food distance
        foodList = newFood.asList()
        foodDistances = [manhattanDistance(newPos, food) for food in foodList]
        minFoodDistance = min(foodDistances) if len(foodDistances) > 0 else 1
        
        # Evaluate the ghost distance
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhostDistance = min(ghostDistances) if len(ghostDistances) > 0 else 1

        # Evaluate the score
        score = successorGameState.getScore()
        
        # Evaluate the number of food left
        numFoodLeft = len(foodList)

        # Evaluate the number of capsules left
        numCapsulesLeft = len(successorGameState.getCapsules())
        
        # Allocate the weights
        foodWeight = 1
        ghostWeight = -0.5
        numFoodWeight = 2
        scoreWeight = 1.5

        # Calculate the evaluation score
        evaluationScore = score * scoreWeight - minFoodDistance * foodWeight - minGhostDistance * ghostWeight - numFoodLeft * numFoodWeight

        return evaluationScore
        

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        根据当前游戏状态使用 self.depth 和 self.evaluationFunction 返回 Minimax 动作。

        实现 Minimax 时，以下是一些可能有用的方法调用:

        gameState.getLegalActions(agentIndex):
        返回 agent 的合法动作列表
        agentIndex=0 表示吃豆人，鬼魂的 agentIndex 大于等于 1

        gameState.generateSuccessor(agentIndex, action):
        返回 agent 执行动作后的后续游戏状态

        gameState.getNumAgents():
        返回游戏中的总 agent 数

        gameState.isWin():
        返回游戏状态是否获胜

        gameState.isLose():
        返回游戏状态是否失败
        """
        def minimax(gameState: GameState, depth: int, agentIndex: int):
            """
            Minimax algorithm
            """
            if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxPacman(gameState, depth, agentIndex)
            else:
                return minGhost(gameState, depth, agentIndex)

        def maxPacman(gameState: GameState, depth: int, agentIndex: int):
            """
            To get the max value of the pacman
            """
            v = float('-inf')  # negative infinity
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                newIndex = (agentIndex + 1) % gameState.getNumAgents()
                v = max(v, minimax(successor, depth + 1, newIndex))  # recursively call minimax
            return v

        def minGhost(gameState: GameState, depth: int, agentIndex: int):
            """
            To get the min value of the ghost
            """
            v = float('inf')
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                newIndex = (agentIndex + 1) % gameState.getNumAgents()
                v = min(v, minimax(successor, depth + 1, newIndex))
            return v

        legalActions = gameState.getLegalActions(0)

        def evaluateAction(action):
            """
            Evaluate the action based on the minimax algorithm
            """
            successorGameState = gameState.generateSuccessor(0, action)
            return minimax(successorGameState, 1, 1)

        bestAction = max(legalActions, key=evaluateAction)

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        global ideal_action

        def alphaBeta(gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
            """
            Alpha-beta algorithm
            """
            if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxPacman(gameState, depth, agentIndex, alpha, beta)
            else:
                return minGhost(gameState, depth, agentIndex, alpha, beta)

        def maxPacman(gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
            """
            To get the max value of the pacman, with alpha-beta pruning
            """
            v = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alphaBeta(successor, depth + 1, (agentIndex + 1) % gameState.getNumAgents(), alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minGhost(gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
            """
            To get the min value of the ghost, with alpha-beta pruning
            """
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphaBeta(successor, depth + 1, (agentIndex + 1) % gameState.getNumAgents(), alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        alpha = float("-inf")
        beta = float("inf")
        max_score = float("-inf")
        for action in gameState.getLegalActions(0):
            score = alphaBeta(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)
            if score > max_score:
                max_score, ideal_action = score, action
            alpha = max(alpha, score)

        return ideal_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(gameState: GameState, depth: int, agentIndex: int):
            """
            Expectimax algorithm
            """
            if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxPacman(gameState, depth, agentIndex)
            else:
                return expectGhost(gameState, depth, agentIndex)

        def maxPacman(gameState: GameState, depth: int, agentIndex: int):
            """
            To get the max value of the pacman
            """
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, expectimax(successor, depth + 1, (agentIndex + 1) % gameState.getNumAgents()))
            return v

        def expectGhost(gameState: GameState, depth: int, agentIndex: int):
            """
            To get the expect value of the ghost
            """
            v = 0
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v += expectimax(successor, depth + 1, (agentIndex + 1) % gameState.getNumAgents())
            return v / len(actions)

        legalActions = gameState.getLegalActions(0)
        bestAction = max(legalActions, key=lambda action: expectimax(gameState.generateSuccessor(0, action), 1, 1))

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # Get necessary information
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Evaluate the food distance
    foodList = newFood.asList()
    foodDistances = [manhattanDistance(newPos, food) for food in foodList]
    minFoodDistance = min(foodDistances) if len(foodDistances) > 0 else 1
    foodScore = 10 / minFoodDistance if minFoodDistance != 0 else 0  # The closer the food, the higher the score

    # Evaluate the ghost distance
    ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    minGhostDistance = min(ghostDistances) if len(ghostDistances) > 0 else 1
    ghostScore = -1 / minGhostDistance if minGhostDistance != 0 else 0  # The closer the ghost, the lower the score

    # Evaluate the scared time
    scaredScore = sum(newScaredTimes)  # The more scared time, the higher the score


    # Evaluate the score
    score = currentGameState.getScore()

    return score + foodScore + ghostScore + scaredScore

# Abbreviation
better = betterEvaluationFunction
