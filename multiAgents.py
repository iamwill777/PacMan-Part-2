# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newGhostStates = successorGameState.getGhostStates()

    "*** YOUR CODE HERE ***"
    # Gets the food closest to ghost
    distance = 0
    foodList = successorGameState.getFood().asList()
    closestDistance = 1000000
    
    # Exits if Pacman runs into ghost or stops
    if action == 'Stop':
        return -closestDistance
    
    for ghost in newGhostStates:
        if ghost.getPosition() == tuple(newPos) and (ghost.scaredTimer == 0):
            return -closestDistance
    
    # Finds closest food and returns the closest distance
    for food in foodList:
        distance = manhattanDistance(food, newPos) + len(foodList) * 100
        if (distance < closestDistance):
            closestDistance = distance
    
    # Makes sure Pacman gets the last food
    if len(foodList) == 0:
        closestDistance = 0

    return closestDistance * -1

def scoreEvaluationFunction(currentGameState):
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

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    # Helper function
    def miniMax(gameState, depth, agent):
        # Increases depth manually
        if agent >= gameState.getNumAgents():
            agent = 0
            depth += 1
        # Exits if it is terminal stage
        if (depth == self.depth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        # If agent is Pacman
        elif (agent == 0):
            return maximum(gameState, depth, agent)
        # If it is ghost
        else:
            return minimum(gameState, depth, agent)
    
    # Find min for ghost
    def minimum(gameState, depth, agent):
        value = ["", 10000000]
        # No more legal actions
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        # Looks through possible actions as long as it's not stop
        for action in gameState.getLegalActions(agent):
            if action != Directions.STOP:
                # Finds minimum value and returns it
                result = miniMax(gameState.generateSuccessor(agent, action), depth, agent+1)
                if type(result) is list:
                    intResult = result[1]
                else:
                    intResult = result
                if intResult < value[1]:
                    value = [action, intResult]
        return value
    # Find max for ghost
    def maximum(gameState, depth, agent):
        value = ["", -10000000]
        # No more legal actions
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        # Looks through possible actions as long as it's not stop    
        for action in gameState.getLegalActions(agent):
            if action != Directions.STOP:
                # Finds maximum value and returns it
                result = miniMax(gameState.generateSuccessor(agent, action), depth, agent+1)
                if type(result) is list:
                    intResult = result[1]
                else:
                    intResult = result
                if intResult > value[1]:
                    value = [action, intResult]                    
        return value
        
    # Gets possible actions for Pacman and returns the maximum one     
    actions = miniMax(gameState, 0, 0)
    return actions[0]
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    # Helper function
    def miniMax(gameState, depth, agent, alpha, beta):
        # Increases depth manually
        if agent >= gameState.getNumAgents():
            agent = 0
            depth += 1
        # Exits if it is terminal stage
        if (depth == self.depth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        # If agent is Pacman
        elif (agent == 0):
            return maximum(gameState, depth, agent, alpha, beta)
        # If it is ghost
        else:
            return minimum(gameState, depth, agent, alpha, beta)
    
    # Find min for ghost
    def minimum(gameState, depth, agent, alpha, beta):
        value = ["", 10000000]
        # No more legal actions
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        # Looks through possible actions as long as it's not stop
        for action in gameState.getLegalActions(agent):
            if action != Directions.STOP:
                # Finds minimum value and returns it while reducing total number of moves
                result = miniMax(gameState.generateSuccessor(agent, action), depth, agent+1, alpha, beta)
                if type(result) is list:
                    intResult = result[1]
                else:
                    intResult = result
                if intResult < value[1]:
                    value = [action, intResult]
                # Reduces total number of moves
                if intResult < alpha:
                    return [action, intResult]
                beta = min(beta, intResult)
        return value
    # Find max for ghost
    def maximum(gameState, depth, agent, alpha, beta):
        value = ["", -10000000]
        # No more legal actions
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        # Looks through possible actions as long as it's not stop    
        for action in gameState.getLegalActions(agent):
            if action != Directions.STOP:
                # Finds maximum value and returns it while reducing total number of moves
                result = miniMax(gameState.generateSuccessor(agent, action), depth, agent+1, alpha, beta)
                if type(result) is list:
                    intResult = result[1]
                else:
                    intResult = result
                if intResult > value[1]:
                    value = [action, intResult]
                # Reduces total number of moves
                if intResult > beta:
                    return [action, intResult]
                alpha = max(alpha, intResult)                    
        return value
        
    # Gets possible actions for Pacman and returns the maximum one     
    actions = miniMax(gameState, 0, 0, -10000000, 10000000)
    return actions[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
