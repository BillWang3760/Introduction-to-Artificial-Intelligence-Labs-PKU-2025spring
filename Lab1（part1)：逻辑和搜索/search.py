# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Stack()  # 构建一个栈
    visited = []             # 创建已访问节点集合
    frontier.push((problem.getStartState(), []))  # 将(初始节点，空动作序列)入栈
    while frontier.isEmpty() == 0:                # 判断栈非空
        state, actions = frontier.pop()           # 从栈中弹出最后一个状态和动作序列
        if problem.isGoalState(state):
            return actions   # 判断是否为目标状态，若是则返回到达该状态的累计动作序列
        if state not in visited:
            visited.append(state)                 # 若当前状态没有访问过，将其标记为已访问
            for next in problem.getSuccessors(state):
                n_state = next[0]
                n_direction = next[1]
                if n_state not in visited:
                    frontier.push((n_state, actions + [n_direction]))  # 子节点按get顺序入栈

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # 构建一个队列
    Frontier = util.Queue()
    # 创建已访问节点集合
    Visited = []
    # 将(初始节点, 空动作序列)入队
    Frontier.push((problem.getStartState(), []))
    #判断队列非空
    while Frontier.isEmpty() == 0:
        # 从队列中弹出一个状态和动作序列
        state, actions = Frontier.pop()
        # 判断是否为目标状态，若是则返回到达该状态的累计动作序列
        if problem.isGoalState(state):
            return actions
        if state not in Visited:
            # 将到达节点标记为已访问节点
            Visited.append(state)
            # 遍历所有后继状态
            for next in problem.getSuccessors(state):
                # 新的后继状态
                n_state = next[0]
                # 新的action
                n_direction = next[1]
                # 若该状态没有访问过
                if n_state not in Visited:
                    # 计算到该状态的动作序列，入队
                    Frontier.push((n_state, actions + [n_direction]))
#! 例题答案如下
# def breadthFirstSearch(problem):
#     """Search the shallowest nodes in the search tree first."""
#     #python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs --frameTime 0
#     #python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5 --frameTime 0
#     "*** YOUR CODE HERE ***"

#     Frontier = util.Queue()
#     Visited = []
#     Frontier.push( (problem.getStartState(), []) )
#     #print 'Start',problem.getStartState()
#     Visited.append( problem.getStartState() )

#     while Frontier.isEmpty() == 0:
#         state, actions = Frontier.pop()
#         if problem.isGoalState(state):
#             #print 'Find Goal'
#             return actions 
#         for next in problem.getSuccessors(state):
#             n_state = next[0]
#             n_direction = next[1]
#             if n_state not in Visited:
                
#                 Frontier.push( (n_state, actions + [n_direction]) )
#                 Visited.append( n_state )

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()      # 构建优先队列
    visited = []                         # 记录已经来过的节点
    frontier.push((problem.getStartState(), []), problem.getCostOfActions([]))
    while frontier.isEmpty() == 0:
        state, actions = frontier.pop()  # 来到当前节点
        if problem.isGoalState(state):
            return actions               # 来到当前节点（即离开优先队列的时候），才能判断是否到达目标
        if state not in visited:
            visited.append(state)
            for next in problem.getSuccessors(state):
                n_state = next[0]
                n_direction = next[1]
                if n_state not in visited:
                    frontier.update((n_state, actions + [n_direction]), problem.getCostOfActions(actions + [n_direction]))
                # 没有考虑过的，直接加入
                # 已在frontier上的节点，若发现其具有更好的代价，更新
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic = nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # 代码主体与UCS相同
    frontier = util.PriorityQueue()      # 构建优先队列
    visited = []                         # 记录已经来过的节点
    startState = problem.getStartState()
    frontier.push((startState, []), problem.getCostOfActions([]) + heuristic(startState, problem))
    while frontier.isEmpty() == 0:
        state, actions = frontier.pop()  # 来到当前节点
        if problem.isGoalState(state):
            return actions               # 来到当前节点（即离开优先队列的时候），才能判断是否到达目标
        if state not in visited:
            visited.append(state)
            for next in problem.getSuccessors(state):
                n_state = next[0]
                n_direction = next[1]
                n_actions = actions + [n_direction]
                if n_state not in visited:
                    frontier.update((n_state, n_actions), problem.getCostOfActions(n_actions) + heuristic(n_state, problem))
                # 没有考虑过的，直接加入
                # 已在frontier上的节点，若发现其具有更好的代价，更新

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
