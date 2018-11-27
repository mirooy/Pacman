# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #V(s) = max_{a in actions} Q(s,a)
     # policy(s) = arg_max_{a in actions} Q(s,a)
        iterations = self.iterations
        for iteration in range(iterations):
          v = self.values.copy()

          for state in self.mdp.getStates():
            states = util.Counter()

            for action in self.mdp.getPossibleActions(state):
              states[action] = self.computeQValueFromValues(state,action)

            v[state] = states[states.argMax()]
          self.values = v


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

        """
        "*** YOUR CODE HERE ***"
        Q_value = 0
        for nextState,prob in self.mdp.getTransitionStatesAndProbs(state, action):
          r = self.mdp.getReward(state,action,nextState)
          Q_value = Q_value + prob * (r + (self.discount * self.values[nextState]))

        return Q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        policy = util.Counter()
        #policy(s) = arg_max_{a in actions} Q(s,a)
        #getQValue(self, state, action):

        for action in self.mdp.getPossibleActions(state):
          policy[action] = self.getQValue(state,action)
        return policy.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        #self.mdp = mdp
        #self.discount = discount
        #self.iterations = iterations
        #self.values = util.Counter() # A Counter is a dict with default 0
        #self.runValueIteration()
        "*** YOUR CODE HERE ***"
     
        iteration_number = 0
        while iteration_number < self.iterations:
          for state in self.mdp.getStates():
            states = util.Counter()

            for action in self.mdp.getPossibleActions(state):
              states[action] = self.computeQValueFromValues(state,action)

            self.values[state] = states[states.argMax()]

            iteration_number = iteration_number + 1
            if iteration_number >= self.iterations:
              return 



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        values = self.values
        priorityqueue = util.PriorityQueue()
        predecessors = {}

        for state in self.mdp.getStates():
          predecessors[state] = set()   

        for state in self.mdp.getStates():
          best_states = util.Counter()

          for action in self.mdp.getPossibleActions(state):
            transition = self.mdp.getTransitionStatesAndProbs(state,action)
            for nextState,prob in transition:
              if prob != 0:
                predecessors[nextState].add(state)
            
            best_states[action] = self.computeQValueFromValues(state,action)

          if not self.mdp.isTerminal(state):
            best_action = best_states[best_states.argMax()]
            diff = abs(values[state] - best_action)
            priorityqueue.update(state,-diff)

        for iteration in range(self.iterations):
          if priorityqueue.isEmpty():
            return
          state = priorityqueue.pop()

          if not self.mdp.isTerminal(state):
            best_states = util.Counter()
            for action in self.mdp.getPossibleActions(state):
              best_states[action] = self.computeQValueFromValues(state,action)

            self.values[state] = best_states[best_states.argMax()]

          for predecessor in predecessors[state]:
            better_p = util.Counter()
            for action in self.mdp.getPossibleActions(predecessor):
              better_p[action] = self.computeQValueFromValues(predecessor,action)
            best_action = better_p[better_p.argMax()]
            diff = abs(values[predecessor] - best_action)

            if diff > self.theta:
              priorityqueue.update(predecessor,-diff)


