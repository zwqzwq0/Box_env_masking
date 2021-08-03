# -*- coding: utf-8 -*-


import gym
from gym import spaces
from gym.spaces import Box,Discrete,MultiDiscrete,Dict
import numpy as np


class Box_Env(gym.Env):
    """
    功能：如下环境封装为 gym env，用于验证强化学习 action masking。

    目标：将地图中的一个箱子移动到目的地

    环境配置参数：二维数组表示地图，1为箱子，0为空地，-1为目的地。示例

    [[0, 0, 1, 0],

    [0, 0, 0, 0],

    [0, -1, 0, 0]]

    观测空间： 当前地图数组

    动作空间：三元组 (col, row, d)，将位于 col、row 的箱子往 d 方向移动一格（0123对应上右下左）。如果 col、row 处无箱子则不行动。

    奖励函数：每走一步reward为 -1。

    """

    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 2
    # }

    def __init__(self, map):
        # mat=mat_dict["mat"]
        self.left_action_embed = np.random.randn(2)
        self.right_action_embed = np.random.randn(2)
        self.up_action_embed = np.random.randn(2)
        self.down_action_embed = np.random.randn(2)
        self.target_col = 0
        self.target_row = 0
        self.start_mat = np.array(map)
        self.map = np.array(map)
        self.h, self.w = self.map.shape
        for i in range(self.h):
            for j in range(self.w):
                if map[i][j] == -1:
                    self.target_col = i
                    self.target_row = j

        #self.observation_space = spaces.Box(-1, 1, shape=(self.h, self.w), dtype=np.int)
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.h,self.w,4)),
            "avail_actions": Box(-10, 10, shape=(self.h,self.w,4, 2)),
            "cart": Box(-1, 1, shape=(self.h, self.w), dtype=np.int),
        })
        # [h,w,4]->(col,row,action)
        self.action_space = spaces.MultiDiscrete([self.h, self.w, 4])

    def update_avail_actions(self):
        self.action_mask=np.zeros(shape=(self.h,self.w,4))
        self.action_assignments=np.zeros(shape=(self.h,self.w,4,2))
        for i in range(self.h):
            for j in range(self.w):          #此处只进行了箱子位置的masking，还可以增加地图边界仅有三个可移动方向的masking
                if self.map[i][j] == 1:
                    self.action_mask[i][j][0] = 1
                    self.action_mask[i][j][1] = 1
                    self.action_mask[i][j][2] = 1
                    self.action_mask[i][j][3] = 1
                    self.action_assignments[i][j][0]=self.up_action_embed
                    self.action_assignments[i][j][1] = self.right_action_embed
                    self.action_assignments[i][j][2] = self.down_action_embed
                    self.action_assignments[i][j][3] = self.left_action_embed

    def reset(self):
        # self.state = np.array([self.start_col,self.start_row])
        self.counts = 0
        self.map = np.copy(self.start_mat)
        #return self.map
        self.update_avail_actions()
        return {
            "action_mask": self.action_mask,
            "avail_actions": self.action_assignments,
            "cart": self.map,
        }

    def step(self, action):
        self.update_avail_actions()
        if self.map[action[0]][action[1]] == 1:  #判断是否有箱子
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

            # 适合多个箱子
            state = np.array([action[0], action[1]])
            action = action[2]

            offsets = [[-1, 0], [0, 1], [1, 0], [0, -1]]
            # print("方向{}".format(offsets[action]))
            col, row = state + offsets[action]

            # h对应x，w对应y
            if self.h > col >= 0 and self.w > row >= 0:  #判断是否可移动
                # 移动前的位置置0
                self.map[state[0]][state[1]] = 0
                # 当前箱子位置置1
                self.map[col][row] = 1

                state = np.array([col, row])
                # self.counts += 1

            now_dist = abs(state[0] - self.target_col) + abs(state[1] - self.target_row)
            done = (now_dist == 0)

            reward = -1  #此处找对箱子位置并移动是否可以reward=1

            # return self.map, reward, done, {}

            return {
            "action_mask": self.action_mask,
            "avail_actions": self.action_assignments,
            "cart": self.map,
            }, reward, done, {}
        reward = -1
        done = False
        #return self.map, reward, done, {}
        return {
            "action_mask": self.action_mask,
            "avail_actions": self.action_assignments,
            "cart": self.map,
            },reward,done,{}

    def render(self, mode='human'):
        return None

    def close(self):
        return None



