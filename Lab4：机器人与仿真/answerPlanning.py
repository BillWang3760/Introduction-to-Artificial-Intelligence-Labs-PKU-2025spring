import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 1
TARGET_THREHOLD = 1
MAXIMUM_CALLS = 3
### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.goal = None
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.calls_for_get_target = 0  # 记录get_target函数的调用次数
        self.current_target_index = 0  # 当前目标节点的下标索引
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      
        # 每次吃掉⼀个⻝物后会调⽤这个函数；提供下⼀个⻝物的位置，你需要规划出⼀条由当前位置前往下⼀个⻝物路径，⼀个由路径上的点组成的列表
        # 每次吃掉一个食物后需要重新规划path路径,current_target_index和calls_for_get_target需要置零
        self.current_target_index = 0
        self.calls_for_get_target = 0
        # 末端优化1：若当前位置和下一个食物之间没有障碍物/距离已经很近，则可以直接前往，否则建树
        if (not self.map.checkline(current_position.tolist(), next_food.tolist())[0])\
                or np.linalg.norm(next_food - current_position) < TARGET_THREHOLD:
            self.path = [next_food]
        ### 你的代码 ###
        # 如有必要，此行可删除
        else:
            self.path = self.build_tree(current_position, next_food)
        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
        # 末端优化2：若当前位置和下一个食物之间没有障碍物/距离已经很近，则可以直接前往
        if (not self.map.checkline(current_position.tolist(), self.path[-1].tolist())[0])\
                or np.linalg.norm(self.path[-1] - current_position) < TARGET_THREHOLD:
            target_pose = self.path[-1]
            return target_pose
        # 若当前节点的执行次数还没有超过n，则目标不变
        if self.calls_for_get_target <= MAXIMUM_CALLS:
            target_pose = self.path[self.current_target_index]
            self.calls_for_get_target += 1
        # 若当前节点的执行次数已经超过n，则将目标改为下一节点
        if self.calls_for_get_target > MAXIMUM_CALLS:
            self.current_target_index += 1  # 将目标改为下一节点
            self.calls_for_get_target = 0   # 该函数调用次数置零
            # 若path数组下标越界，则重新规划一条到终点的路径
            if self.current_target_index >= len(self.path):
                self.find_path(current_position, self.path[-1])
                self.current_target_index = 0  # 重新规划路径后产生新的路径数组，旧索引已失效，新索引需要置零
            target_pose = self.path[self.current_target_index]
        ### 你的代码 ###
        return target_pose
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        path = []
        graph: List[TreeNode] = []
        graph.append(TreeNode(-1, start[0], start[1]))
        ### 你的代码 ###
        x_min, y_min = np.min(self.walls, axis=0)
        x_max, y_max = np.max(self.walls, axis=0)
        while True:
            # 步骤1. 在地图上随机采样一个点
            random_point = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ])
            # 步骤2.计算RRT树上离随机点最近的点
            nearest_idx, nearest_distance = self.find_nearest_point(random_point, graph)
            nearest_node = graph[nearest_idx]
            # 步骤3. 以nearest_point为起点，以nearest_point → random_point射线为方向，前进STEP_DISTANCE的距离
            is_empty, newpoint = self.connect_a_to_b(nearest_node.pos, random_point)
            # 步骤4. 检查步骤3中生成路径的碰撞状态is_empty，如果没有碰撞到障碍物，加入到RRT树中
            if is_empty:
                new_node = TreeNode(nearest_idx, newpoint[0], newpoint[1])  # 新节点，父节点为nearest_idx
                graph.append(new_node)  # 将新节点加入RRT
                # 步骤5. 检查是否到达目标点或者和目标点足够接近
                if np.linalg.norm(goal - newpoint) < TARGET_THREHOLD:
                    # 回溯构造规划路径path
                    path.append(goal)  # 首先加入目标点
                    current_node = new_node
                    while current_node.parent_idx != -1:  # 沿着父节点回溯到根节点
                        path.append(current_node.pos)
                        current_node = graph[current_node.parent_idx]
                    path.append(current_node.pos)  # 最后加入起点
                    path.reverse()  # 反转路径，得到从起点到目标点的路径
                    break
        ### 你的代码 ###
        return path

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = 10000000.
        ### 你的代码 ###
        index = 0
        for node in graph:
            distance = np.linalg.norm(point - node.pos)
            if distance < nearest_distance:
                nearest_idx = index
                nearest_distance = distance
            index += 1
        ### 你的代码 ###
        return nearest_idx, nearest_distance
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        is_empty = False
        newpoint = np.zeros(2)
        ### 你的代码 ###
        newpoint = point_a + STEP_DISTANCE * (point_b - point_a) / np.linalg.norm(point_b - point_a)
        is_empty = not self.map.checkline(point_a.tolist(), newpoint.tolist())[0]
        ### 你的代码 ###
        return is_empty, newpoint
