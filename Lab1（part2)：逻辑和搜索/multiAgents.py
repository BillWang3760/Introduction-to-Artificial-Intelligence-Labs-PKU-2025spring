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
# Revised by TAs from Intro to AI class (2025 Spring) of PKU.


from util import manhattanDistance
from game import Directions
import random, util
from math import sqrt, log

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

        return successorGameState.getScore()

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
      Your minimax agent (question 1)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          GameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          GameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          GameState.getNumAgents():
            Returns the total number of agents in the game
          GameState.isWin(), GameState.isLose():
            Returns whether or not the game state is a terminal state
        """
        def maximizer(state, depth, index_of_agent):
            maxiAction = None
            # condition for termination of recursive method calls
            def terminal_condition(state, depth):
                if state.isWin() or state.isLose() or depth == 0:
                    return True
                # GameState.isWin(), GameState.isLose(): Returns whether or not the game state is a terminal state
                # 初始传入给maximizer的depth参数为self.depth，递归调用过程中depth参数依次递减，最终在遍历minimax搜索树的叶子节点之后depth减为0
                else:
                    return False
            if terminal_condition(state,depth) == True:
                return (self.evaluationFunction(state), None)
            # initialize value
            value = - 1 << 31
            # for every legal action, update value and maxiAction
            # GameState.getLegalActions(agentIndex): Returns a list of legal actions for an agent
            legalActions = state.getLegalActions(index_of_agent)
            for legalAction in legalActions:
                #  GameState.generateSuccessor(agentIndex, action): Returns the successor game state after an agent takes an action
                successorGameState = state.generateSuccessor(index_of_agent, legalAction)
                valueOfSuccessor = minimizer(successorGameState, depth, index_of_agent + 1)[0]
                # 递归调用minimizer函数获取valueOfSuccessor， 注意此时是在同一搜索层内递归，故参数depth在此处保持不变
                # 但行动的智能体由Pacman变为下一个幽灵，故index_of_agent需要递增
                if value < valueOfSuccessor:
                    value = valueOfSuccessor
                    maxiAction = legalAction  # 注意不要错误地将子节点的动作作为当前层的最佳动作返回，应记录当前层的合法动作
                    # update value and maxiAction
            return (value, maxiAction)

        def minimizer(state, depth, index_of_agent):
            miniAction = None
            def terminal_condition(state,depth):
                if state.isWin() or state.isLose() or depth == 0:
                    return True
                # 注释见maximizer中的line142~143
                else:
                    return False
            if terminal_condition(state, depth) == True:
                return (self.evaluationFunction(state), miniAction)
            # initialize value
            value = 1 << 31
            # for every legal action, update value and miniAction
            legalActions = state.getLegalActions(index_of_agent)
            for legalAction in legalActions:
                successorGameState = state.generateSuccessor(index_of_agent, legalAction)
                #  GameState.getNumAgents(): Returns the total number of agents in the game
                numAgents = state.getNumAgents()
                # 若已经来到了当前搜索层的底部，则需要对下一个搜索层的顶部进行递归调用
                if index_of_agent == numAgents - 1:
                    valueOfSuccessor = maximizer(successorGameState, depth - 1, 0)[0]
                    # 递归调用maximizer函数获取valueOfSuccessor，注意此时是已不在同一搜索层内，故参数depth需要递减
                    # agentIndex = 0 means Pacman
                    if value > valueOfSuccessor:
                        value = valueOfSuccessor
                        miniAction = legalAction  # 注意不要错误地将子节点的动作作为当前层的最佳动作返回，应记录当前层的合法动作
                else:
                    valueOfSuccessor = minimizer(successorGameState, depth, index_of_agent + 1)[0]
                    # 递归调用minimizer函数获取返回元组， 注意此时是在同一搜索层内递归，故参数depth在此处保持不变
                    if value > valueOfSuccessor:
                        value = valueOfSuccessor
                        miniAction = legalAction  # 注意不要错误地将子节点的动作作为当前层的最佳动作返回，应记录当前层的合法动作
            return (value, miniAction)      
        action = maximizer(gameState, self.depth, 0)[1]
        return action 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maximizer(state, depth, index_of_agent, alpha, beta):
            maxiAction = None
            #condition for termination of recursive method calls
            def terminal_condition(state, depth):
                if state.isWin() or state.isLose() or depth == 0:
                    return True
                else:
                    return False
            if terminal_condition(state,depth) == True:
                return (self.evaluationFunction(state), None)
            # initialize value
            value = - 1 << 31
            # for every legal action, update value, maxiAction and alpha:
            legalActions = state.getLegalActions(index_of_agent)
            for legalAction in legalActions:
                successorGameState = state.generateSuccessor(index_of_agent, legalAction)
                valueOfSuccessor = minimizer(successorGameState, depth, index_of_agent + 1, alpha, beta)[0]
                if value < valueOfSuccessor:
                    value = valueOfSuccessor
                    maxiAction = legalAction
                # Alpha-Beta剪枝
                if value > beta:
                    return (value, None)  # 注意此处应返回一个元组
                alpha = max(alpha, value)
            return (value, maxiAction)
        
        def minimizer(state, depth, index_of_agent, alpha, beta):
            miniAction = None
            def terminal_condition(state,depth):
                if state.isWin() or state.isLose() or depth == 0:
                    return True
                else:
                    return False
            if terminal_condition(state, depth) == True:
                return (self.evaluationFunction(state), miniAction)
            # initialize value
            value = 1 << 31
            # for every legal action, update value, miniAction and beta
            legalActions = state.getLegalActions(index_of_agent)
            for legalAction in legalActions:
                successorGameState = state.generateSuccessor(index_of_agent, legalAction)
                numAgents = state.getNumAgents()
                if index_of_agent == numAgents - 1:
                    valueOfSuccessor = maximizer(successorGameState, depth - 1, 0, alpha, beta)[0]
                    if value > valueOfSuccessor:
                        value = valueOfSuccessor
                        miniAction = legalAction
                else:
                    valueOfSuccessor = minimizer(successorGameState, depth, index_of_agent + 1, alpha, beta)[0]
                    if value > valueOfSuccessor:
                        value = valueOfSuccessor
                        miniAction = legalAction
                # Alpha-Beta剪枝
                if value < alpha:
                    return (value, None)  # 注意此处应返回一个元组
                beta = min(beta, value)
            return (value, miniAction) 
        # initialize alpha/beta
        alpha = - 1 << 31
        beta = 1 << 31
        action = maximizer(gameState, self.depth, 0, alpha, beta)[1]
        return action 


class MCTSAgent(MultiAgentSearchAgent):

    def getAction(self, gameState, mcts_time_limit=10):

        class Node:

            def __init__(self, data):
                self.north = None                   # 选择当前action为“north”对应的节点, <class 'Node'>
                self.east = None                    # 选择当前action为“east”对应的节点, <class 'Node'>
                self.west = None                    # 选择当前action为“west”对应的节点, <class 'Node'>
                self.south = None                   # 选择当前action为“south”对应的节点, <class 'Node'>
                self.stop = None                    # 选择当前action为“stop”对应的节点, <class 'Node'>
                self.parent = None                  # 父节点, <class 'Node'>
                self.statevalue = data[0]           # 该节点对应的游戏状态, <class 'GameState' (defined in pacman.py)>
                self.numerator = data[1]            # 该节点的分数
                self.denominator = data[2]          # 该节点的访问次数

        def Selection(cgs, cgstree):
            '''
                cgs: current game state, <class 'GameState' (defined in pacman.py)>
                cgstree: current game state tree, <class 'Node'>
                
                YOUR CORE HERE (~30 lines or fewer)
                1. You have to find a node that is not completely expanded (e.g., node.north is None)
                2. When you find the node, return its corresponding game state and the node itself.
                3. You should use best_UCT() to find the best child of a node each time.

            '''
            # 注意，Selection函数与课件中的SELECTACTION函数不同，其所对应的应为TREEPOLICY函数
            while True:
                legal_actions = cgstree.statevalue.getLegalActions(0)
                # getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
                # 1. find a node that is not completely expanded (e.g., node.north is None)
                if ("North" in legal_actions and cgstree.north == None) or \
                    ("South" in legal_actions and cgstree.south == None) or \
                    ("West" in legal_actions and cgstree.west == None) or \
                    ("East" in legal_actions and cgstree.east == None) or \
                    ("Stop" in legal_actions and cgstree.stop == None) or \
                    len(legal_actions) == 0:  # in case that the node is completely unexpanded
                        break  # 2. When the node is found, return its corresponding game state and the node itself.
                # children: list of tuples, each tuple contains a child node and the action that leads to it
                children = []
                if "North" in legal_actions:
                    children.append((cgstree.north, "North"))
                if "South" in legal_actions:
                    children.append((cgstree.south, "South"))
                if "West" in legal_actions:
                    children.append((cgstree.west, "West"))
                if "East" in legal_actions:
                    children.append((cgstree.east, "East"))
                if "Stop" in legal_actions:
                    children.append((cgstree.stop, "Stop"))
                if len(children) == 0:
                    break  # 防止children列表为空
                bestNodeState, bestAction = best_UCT(children, random_prob=0.3)
                # return (children[max_index][0].statevalue, children[max_index][1])
                cgs = bestNodeState
                if bestAction == "North":
                    cgstree = cgstree.north
                elif bestAction == "South":
                    cgstree = cgstree.south
                elif bestAction == "West":
                    cgstree = cgstree.west
                elif bestAction == "East":
                    cgstree = cgstree.east
                elif bestAction == "Stop":
                    cgstree = cgstree.stop
            return (cgs, cgstree)

        def Expansion(cgstree):
            legal_actions = cgstree.statevalue.getLegalActions(0)
            '''
                YOUR CORE HERE (~20 lines or fewer)
                1. You should expand the current game state tree node by adding all of its children.
                2. You should use Node() to create a new node for each child.
                3. You can traverse the legal_actions to find all the children of the current game state tree node.
            '''
            for legal_action in legal_actions:
                data = []
                data.append(cgstree.statevalue.generateSuccessor(0, legal_action))  # self.statevalue = data[0]  该节点对应的游戏状态, <class 'GameState' (defined in pacman.py)>
                data.append(0)  # self.numerator = data[1]   该节点的分数  simulation_score = 0
                data.append(1)  # self.denominator = data[2] 该节点的访问次数，初始化为1防止除数为0 (Why?)
                if legal_action == "North" and cgstree.north == None:
                    cgstree.north = Node(data)      # def __init__(self, data):
                    cgstree.north.parent = cgstree  # 注意记得设置扩展子节点的父节点，以备Backpropagation时使用
                elif legal_action == "South" and cgstree.south == None:
                    cgstree.south = Node(data)
                    cgstree.south.parent = cgstree
                elif legal_action == "West" and cgstree.west == None:
                    cgstree.west = Node(data)
                    cgstree.west.parent = cgstree
                elif legal_action == "East" and cgstree.east == None:
                    cgstree.east = Node(data)
                    cgstree.east.parent = cgstree
                elif legal_action == "Stop" and cgstree.stop == None:
                    cgstree.stop = Node(data)
                    cgstree.stop.parent = cgstree

        def Simulation(cgs, cgstree):
            '''
                This implementation is different from the one taught during the lecture.
                All the nodes during a simulation trajectory are expanded.
                We choose to more quickly expand our game tree (and hence pay more memory) to get a faster MCTS improvement in return.
            '''
            simulation_score = 0
            while cgstree.statevalue.isWin() is False and cgstree.statevalue.isLose() is False:
                cgs, cgstree = Selection(cgs, cgstree)
                Expansion(cgstree)
            '''
                YOUR CORE HERE (~4 lines)
                You should modify the simulation_score of the current game state.
            '''
            if cgstree.statevalue.isWin():
                simulation_score += 1
            elif cgstree.statevalue.isLose():
                simulation_score += 0
            return simulation_score, cgstree

        def Backpropagation(cgstree, simulation_score):
            while cgstree.parent is not None:
                '''
                    YOUR CORE HERE (~3 lines)
                    You should recursively update the numerator and denominator of the game states until you reaches the root of the tree.
                '''
                cgstree.numerator += simulation_score
                cgstree.denominator += 1
                cgstree = cgstree.parent
            # 如下代码会导致Pacman在地图中仅剩最后一颗豆子时出现抖动现象：
            # while cgstree.parent is not None:
            #     cgstree.numerator += simulation_score
            #     cgstree.denominator += 1
            #     cgstree = cgstree.parent
            # cgstree.numerator += simulation_score
            # cgstree.denominator += 1
            # 尝试解释如下：
            # 实际代码：因忽略根节点更新，UCT探索项(Exploration)项消失，算法快速收敛到当前最优路径，行为稳定(但可能陷入局部最优)
            # 注释代码：因更新根节点，探索项随模拟次数动态放大，导致UCT值波动和决策抖动，尤其在长期博弈中更明显
            return cgstree

        # 根据UCT算法选择最好的子节点及其对应的action。你不需要修改这个函数。
        def best_UCT(children, random_prob=0.3):
            '''
                children: list of tuples, each tuple contains a child node and the action that leads to it
                random_prob: the probability of choosing a random action when UCT values are the same

                return: the best child node's game state and the action that leads to it
            '''
            i = 0
            while i < len(children):
                if children[i][0] is None or children[i][1] == 'Stop':
                    children.pop(i)
                else:
                    i = i+1

            children_UCT = []
            for i in range(len(children)):
                
                value = ((children[i][0].numerator / children[i][0].denominator) + sqrt(2) * sqrt(
                    ((log(children[i][0].parent.denominator))/log(2.71828)) / children[i][0].denominator)), children[i][1]

                children_UCT.append(value)

            max_index = 0
            equal_counter = 1

            for i in range(len(children_UCT)-1):
                if children_UCT[i][0] == children_UCT[i+1][0]:
                    equal_counter = equal_counter + 1
            
            # 如果所有的UCT值都相等，用启发式函数来选择
            if equal_counter == len(children_UCT):
                
                # 有random_prob的概率随机选择
                decision_maker = random.randint(1, 101)
                if decision_maker < (1 - random_prob) * 100:
                    eval_list = []
                    max_index_list = []
                    for i in range(len(children)):
                        eval_list.append(HeuristicFunction(
                            children[i][0].statevalue))
                    max_index_list.append(eval_list.index(max(eval_list)))
                    maxval = eval_list.pop(max_index_list[-1])
                    eval_list.insert(max_index_list[-1], -9999)
                    while maxval in eval_list:
                        max_index_list.append(eval_list.index(max(eval_list)))
                        eval_list.pop(max_index_list[-1])
                        eval_list.insert(max_index_list[-1], -9999)
                    max_index = random.choice(max_index_list)
                else:
                    max_index = random.randint(0, len(children)-1)
            
            # 否则选最好的UCT对应的节点
            else:
                maximumvalueofUCT = -9999
                for i in range(len(children_UCT)):
                    if children_UCT[i][0] > maximumvalueofUCT:
                        max_index = i
                        maximumvalueofUCT = children_UCT[i][0]
            return (children[max_index][0].statevalue, children[max_index][1])

        # 样例启发式函数，你不需要修改。这个函数会返回一个游戏状态的分数。
        def HeuristicFunction(currentGameState):
            new_position = currentGameState.getPacmanPosition()
            new_food = currentGameState.getFood().asList()

            food_distance_min = float('inf')
            for food in new_food:
                food_distance_min = min(
                    food_distance_min, manhattanDistance(new_position, food))

            ghost_distance = 0
            ghost_positions = currentGameState.getGhostPositions()

            for i in ghost_positions:
                ghost_distance = manhattanDistance(new_position, i)
                if (ghost_distance < 1):
                    return -float('inf')

            food = currentGameState.getNumFood()
            pellet = len(currentGameState.getCapsules())

            food_coefficient = 999999
            pellet_coefficient = 19999
            food_distance_coefficient = 999

            game_rewards = 0
            if currentGameState.isLose():
                game_rewards = game_rewards - 99999
            elif currentGameState.isWin():
                game_rewards = game_rewards + 99999

            answer = (1.0 / (food + 1) * food_coefficient) + ghost_distance + (
                1.0 / (food_distance_min + 1) * food_distance_coefficient) + (
                1.0 / (pellet + 1) * pellet_coefficient) + game_rewards

            return answer

        def endSelection(cgstree):
            children = []
            destin = (cgstree.north, "North")
            children.append(destin)
            destin = (cgstree.east, "East")
            children.append(destin)
            destin = (cgstree.south, "South")
            children.append(destin)
            destin = (cgstree.west, "West")
            children.append(destin)
            destin = (cgstree.stop, "Stop")
            children.append(destin)
            action = best_UCT(children, random_prob=0.0)[1]
            return action
        
        '''
            YOUR CODE HERE (~1-2 line)
            initialize root node cgstree (class Node)
        '''
        data = []
        data.append(gameState)   # self.statevalue = data[0]  该节点对应的游戏状态, <class 'GameState' (defined in pacman.py)>
        data.append(0)           # self.numerator = data[1]   该节点的分数
        data.append(1)           # self.denominator = data[2] 该节点的访问次数
        cgstree = Node(data)     # def __init__(self, data):

        for _ in range(mcts_time_limit):
            gameState, cgstree = Selection(gameState, cgstree)                  # 根据当前的游戏状态和搜索树，选择一个最好的子节点
            Expansion(cgstree)                                                  # 扩展这个选到的节点
            simulation_score, cgstree = Simulation(gameState, cgstree)          # 从这个节点开始模拟
            cgstree = Backpropagation(cgstree, simulation_score)                # 将模拟的结果回溯到根节点，cgstree为根节点
            gameState = cgstree.statevalue                              
        
        return endSelection(cgstree)
