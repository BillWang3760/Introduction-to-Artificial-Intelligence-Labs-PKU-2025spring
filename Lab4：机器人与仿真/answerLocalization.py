from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 0.7
MAX_ERROR = 50000

K = 0.4
NOISE_POSITION_STD = 0.1
NOISE_THETA_STD = 0.1


def is_valid_particle(x, y, walls):
    for wall in walls:
        wall_x, wall_y = wall
        distance = np.sqrt((x - wall_x)**2 + (y - wall_y)**2)
        if distance < COLLISION_DISTANCE:
            return False
    return True


def add_noise(particle, walls):
    while True:
        new_position = particle.position + np.random.normal(0, NOISE_POSITION_STD, size=2)
        new_theta = (particle.theta + np.random.normal(0, NOISE_THETA_STD)) % (2 * np.pi)
        if is_valid_particle(new_position[0], new_position[1], walls):
            return Particle(new_position[0], new_position[1], new_theta, particle.weight)


### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    e.g. walls = np.array([
             [1.0, 3.0],  # 墙壁1的坐标(x=1.0, y=3.0)
             [4.0, 2.0],  # 墙壁2的坐标(x=4.0, y=2.0)
             [0.0, 5.0],  # 墙壁3的坐标(x=0.0, y=5.0)
         ])
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    ### 你的代码 ###
    x_min, y_min = np.min(walls, axis=0)
    x_max, y_max = np.max(walls, axis=0)
    count = 0
    while count < N:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        if is_valid_particle(x, y, walls):
            theta = np.random.uniform(0, 2 * np.pi)
            all_particles.append(Particle(x, y, theta, 1.0 / N))
            count += 1
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    weight = np.exp(-K * np.linalg.norm(estimated - gt))  # np.linalg.norm: 用于计算向量/矩阵的范数
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    ### 你的代码 ###
    N = len(particles)  # 总采样点数量
    weights = np.array([particle.weight for particle in particles])  # 每个粒子的权重
    sample_num = np.floor(weights * N).astype(int)  # 将粒子总数乘以每个粒子的权重，并在每个粒子周围采样该数量的点
    for index, particle in enumerate(particles):
        count = 0
        while count < sample_num[index]:
            resampled_particles.append(add_noise(particle, walls))
            count += 1
    rest_num = N - np.sum(sample_num)  # 剩余的位置⽤均匀采样补全粒⼦总数
    rest_particles = generate_uniform_particles(walls, rest_num)
    resampled_particles += rest_particles
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta += dtheta
    p.position[0] += (traveled_distance * np.cos(p.theta))
    p.position[1] += (traveled_distance * np.sin(p.theta))
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    weights = np.array([particle.weight for particle in particles])
    final_result = particles[np.argmax(weights)]
    ### 你的代码 ###
    return final_result