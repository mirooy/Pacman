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

        score = successorGameState.getScore()
        ghost_score = 2
        food_score = 10
        for ghost in newGhostStates:
          distance_ghost = manhattanDistance(newPos, ghost.getPosition())
          if distance_ghost>0:
            score = score- ghost_score/distance_ghost
         
        distance_food = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(distance_food) !=0:
          shortest_food = min(distance_food)
          score = score + food_score/shortest_food
          #print score
        return score

          #score = score - distance_ghost

        #distance_food = 0
        #for food in newFood:
         # distance_food = manhattanDistance(newPos,food)
         # score = score - distance_ghos


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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(state,depth,agent):
          if agent == state.getNumAgents():
            return minimax(state,depth+1,0)
          if depth == self.depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
          ghosts = list()
          for action in state.getLegalActions(agent):
            ghosts.append(minimax(state.generateSuccessor(agent,action),depth,agent + 1))
          if agent ==0:
            return max(ghosts)
          else:
            return min(ghosts)

        return max(gameState.getLegalActions(0),
          key = lambda x: minimax(gameState.generateSuccessor(0,x),0,1))
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        beta = float('inf')
        alpha = -float('inf')
        v = -float('inf')
        direction = gameState.getLegalActions()[0]
        preferred = None
      
        for action in gameState.getLegalActions(0):
          next_= gameState.generateSuccessor(0,action)
          finalscore = self.minizer(next_,0,1,alpha,beta)
          if finalscore> v:
            v = finalscore
            direction = action
      
          alpha = max(alpha,finalscore)
        return direction
    def maximizer(self,gameState,depth,alpha,beta):
        v = -float('inf')
        if self.depth == depth:
          return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
          return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(0):
          next_=gameState.generateSuccessor(0,action)
          v = max(v, self.minizer(next_,depth,1,alpha,beta))
          
          if v > beta:
            return v
          alpha = max(alpha,v)
        return v

    def minizer(self,gameState,depth,agent,alpha,beta):
        v = float('inf')
        if self.depth == depth:
          return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
          return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agent):
          next_ = gameState.generateSuccessor(agent,action)
          if agent <gameState.getNumAgents() - 1:
            v = min(v, self.minizer(next_,depth,agent+1,alpha,beta))
          elif agent == gameState.getNumAgents() - 1:
            v = min(v,self.maximizer(next_,depth+1,alpha,beta))

          if v < alpha:
              return v
          beta = min(beta,v)
          
        return v


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
        def expected(state,depth,agent):
          if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if depth == self.depth:
            return self.evaluationFunction(state)
          if agent == state.getNumAgents():
            return expected(state,depth+1,0)
          ghosts = list()
          for action in state.getLegalActions(agent):
            ghosts.append(expected(state.generateSuccessor(agent,action),depth,agent + 1))
          if agent == 0:
            return max(ghosts)
          else:
            return sum(ghosts)/len(ghosts)
        return max(gameState.getLegalActions(0),
          key = lambda x : expected(gameState.generateSuccessor(0,x),0,1))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghost_score = 2
    food_score = 2
    capsules = 2

    score = currentGameState.getScore()
    
    for ghost in newGhostStates:
      distance_ghost = manhattanDistance(newPos, ghost.getPosition())
      if distance_ghost>0:
        if ghost.scaredTimer > 0 :
          score = score+ capsules/distance_ghost
        else:
          score = score -ghost_score/distance_ghost
         
    distance_food = [manhattanDistance(newPos, food) for food in newFood.asList()]
    if len(distance_food) !=0:
      shortest_food = min(distance_food)
      score = score + food_score/shortest_food
          #print score
    return score


    

# Abbreviation
better = betterEvaluationFunction

