from agent.base_agent import Base_agent
import param
import random
import gym
import numpy as np
import math


class Red_agent(Base_agent):
    """
    已训练好的智能体封装代码
    """
    def __init__(self, control_num, side, agent_config):
        super().__init__(control_num, side, agent_config)
        self.frame_feature_size = agent_config['frame_feature_size'] if 'frame_feature_size' in agent_config.keys() else 10
        self.cur_process_obs = [0 for i in range(self.frame_feature_size)]
        # 自定义的智能体
        self.trainable_agent = None
        # self.trainable_agent.load_state_dict(torch.load(agent_config['checkpoint']))

    def agent_step(self, obs):
        self.obs_tot = obs
        self.postprocess_obs()
        action = self.trainable_agent.compute_action(self.cur_process_obs)
        action_process = self.postprocess_action(action)
        return action_process

    def postprocess_obs(self):
        """
        Returns:
            postprocessed obs
        """
        obs_control_side = self._common_attribute_state_process()
        obs_control_side = np.array(obs_control_side).clip(-10, 10)
        self.cur_process_obs = obs_control_side
        return obs_control_side

    def postprocess_action(self, action):
        """
        根据不同的输入形式，选用不同的动作处理方式
        """
        return self.input_index_action(action)

    def input_index_action(self, action):
        #####################################################################
        # 自身动作的编码
        #####################################################################
        action = int(action)
        switch_missile_action = int(action // 162)
        action %= 162
        launch_missile_action = int(action // 81)
        action %= 81
        action_ce = action // 27 - 1
        action %= 27
        action_ca = action // 9 - 1
        action %= 9
        action_cr = action - 1
        action %= 3
        action_cT = action * 0.5
        #####################################################################
        # 对手策略的生成
        #####################################################################
        control_side = 'red'
        action_input = dict()
        action_input[control_side] = {
            f'{control_side}_0': {
                'mode': 0,
                "fcs/aileron-cmd-norm": action_ca,
                "fcs/rudder-cmd-norm": action_cr,
                "fcs/elevator-cmd-norm": action_ce,
                "fcs/throttle-cmd-norm": action_cT,
                "fcs/weapon-launch": launch_missile_action,
                "switch-missile": switch_missile_action,
                "change-target": 9,
            }}
        return action_input

    def _common_attribute_state_process(self):
        """
        :return: 返回状态编码信息
        """
        control_side = self.control_side
        alley = ['base_state']
        post_process_obs_control_side = []
        post_process_obs_control_side_dict = self._single_player_state_process(flag=control_side)
        for key in alley:
            post_process_obs_control_side.extend(post_process_obs_control_side_dict[key])
        return post_process_obs_control_side

    def _single_player_state_process(self, flag):
        """
        :return: 返回编码的状态信息
        """
        post_process_obs = dict()
        post_process_obs['base_state'] = self._state_process(flag=flag)
        return post_process_obs

    def _state_process(self, flag='red'):
        """
        TODO：本函数是状态编码信息，此处只是编码了生命值和高度信息
        TODO：编程开发者可以参考智空文档，自行定义智能体训练需要的观测信息
        """
        if self.is_done[flag]:
            return [0 for i in range(23)]
        else:
            post_process_obs = []
            obs = self.obs_tot[f'{flag}_0']
            # 生命值和高度
            post_process_obs.append(obs['LifeCurrent'])
            post_process_obs.append(obs['position/h-sl-ft'])
            # 经纬度
            post_process_obs.append(obs['position/long-gc-deg'])
            post_process_obs.append(obs['position/lat-geod-deg'])
            # 姿态
            post_process_obs.append(obs['attitude/pitch-rad'])
            post_process_obs.append(obs['attitude/roll-rad'])
            post_process_obs.append(obs['attitude/psi-deg'])
            post_process_obs.append(obs['aero/beta-deg'])
            # 速度
            post_process_obs.append(obs['velocities/u-fps'])
            post_process_obs.append(obs['velocities/v-fps'])
            post_process_obs.append(obs['velocities/w-fps'])
            # 角速度
            post_process_obs.append(obs['velocities/p-rad_sec'])
            post_process_obs.append(obs['velocities/q-rad_sec'])
            post_process_obs.append(obs['velocities/r-rad_sec'])
            # 武器信息
            post_process_obs.append(obs['SRAAMCurrentNum'])
            post_process_obs.append(obs['AMRAAMCurrentNum'])
            post_process_obs.append(obs['SRAAM1_CanReload'])
            post_process_obs.append(obs['SRAAM2_CanReload'])
            post_process_obs.append(obs['AMRAAM1_CanReload'])
            post_process_obs.append(obs['AMRAAM2_CanReload'])
            post_process_obs.append(obs['AMRAAM3_CanReload'])
            post_process_obs.append(obs['AMRAAM4_CanReload'])
            post_process_obs.append(obs['MissileAlert'])
            return post_process_obs