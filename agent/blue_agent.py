from agent.base_agent import Base_agent
import param
import random
import gym
import numpy as np
import math


class Blue_agent(Base_agent):
    def __init__(self, control_num, side, agent_config):
        super().__init__(control_num, side, agent_config)

    def agent_step(self, obs):
        """
        TODO: 在此处可以定义对手的动作，下面的例子是对手采用随机策略，但是不发射武器
        """
        opponent_action = {
            f'blue_0': {
                'mode': 0,
                "fcs/aileron-cmd-norm": 2 * random.random() - 1,
                "fcs/rudder-cmd-norm": 2 * random.random() - 1,
                "fcs/elevator-cmd-norm": 2 * random.random() - 1,
                "fcs/throttle-cmd-norm": random.random(),
                "fcs/weapon-launch": 0,
                "switch-missile": 0,
                "change-target": 9,
            }}
        return opponent_action

