from custom_env.env_zhikong import Base_env
import param
import random
import gym
import numpy as np
import math


class DogFight(Base_env):
    def __init__(self, config=None, render=0):
        Base_env.__init__(self, config, render)
        self.collect_num = 0
        self.cur_process_obs = [0 for i in range(self.frame_feature_size)]
        self.cur_process_obs_opponent = [0 for i in range(self.frame_feature_size)]

    def reset_var(self):
        self.collect_num += 1
        self.cur_process_obs = [0 for i in range(self.frame_feature_size)]
        self.cur_process_obs_opponent = [0 for i in range(self.frame_feature_size)]

    def set_s_a_space(self):
        """
        define observation space and action space
        观测空间是一个Box空间，包含了环境状态的连续值。
        动作空间是一个Discrete空间，包含了108个离散的动作。
        """
        self.observation_space = gym.spaces.Box(low=-10, high=10, dtype=np.float32,
                                                shape=(self.frame_feature_size,))
        self.action_space = gym.spaces.Discrete(108)

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
        switch_acm_action = int(action  // 324)    #新增切换acm模式
        action %= 324
        switch_missile_action = int(action // 162)
        action %= 162
        launch_missile_action = int(action // 81)
        action %= 81
        action_ce = action // 27 - 1
        action %= 27
        action_ca = action // 9 - 1
        action %= 9
        action_cr = action // 3 - 1
        action %= 3
        action_cT = action * 0.5
        #####################################################################
        # 对手策略的生成
        #####################################################################
        control_side = self.control_side
        opponent_side = self.opponent_side
        action_input = dict()
        action_input[opponent_side] = self.get_opponent_action()
        action_input[control_side] = {
            f'{control_side}_0': {
                'mode': 0,
                "fcs/aileron-cmd-norm": action_ca,
                "fcs/rudder-cmd-norm": action_cr,
                "fcs/elevator-cmd-norm": action_ce,
                "fcs/throttle-cmd-norm": action_cT,
                "fcs/weapon-launch": launch_missile_action,
                #新增切换acm模式
                "switch-acmType":switch_acm_action,
                "switch-missile": switch_missile_action,
                "change-target": 9,
            }}
        return action_input

    def get_opponent_action(self):
        """
        TODO: 在此处可以定义对手的动作，下面的例子是给予了一个平飞的操作，对手采用平飞策略，但是不发射武器
        """
        opponent_side = self.opponent_side
        obs = self.obs_tot[opponent_side][f'{opponent_side}_0']
        control_side_psi, control_side_v, control_side_h = \
            obs['attitude/psi-deg'], \
            math.sqrt(obs['velocities/u-fps'] ** 2 +
                      obs['velocities/v-fps'] ** 2 +
                      obs['velocities/w-fps'] ** 2), \
            obs['position/h-sl-ft']
        opponent_action = {
            f'{opponent_side}_0': {
                'mode': 0,
                "fcs/aileron-cmd-norm": random.random() * 2 - 1,
                "fcs/rudder-cmd-norm": random.random() * 2 - 1,
                "fcs/elevator-cmd-norm": random.random() * 2 - 1,
                "fcs/throttle-cmd-norm": random.random() * 0.5,
                "fcs/weapon-launch": 1,
                "switch-missile": 0,
                "change-target": 9,
            }}
        return opponent_action

    def postprocess_obs(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can normalize obs or postprocess obs here
                 用于对观测值进行后处理
        Returns:
            postprocessed obs
            在这里，它对观测值进行了裁剪并返回了控制方（control_side）的观测值
        """
        obs_control_side, obs_opponent_side = self._common_attribute_state_process()
        obs_control_side, obs_opponent_side = np.array(obs_control_side).clip(-10, 10), np.array(obs_opponent_side).clip(-10, 10)
        self.cur_process_obs_opponent = obs_opponent_side
        self.cur_process_obs = obs_control_side
        return obs_control_side

    # 用于初始化飞机的位置和速度等信息。这个方法随机生成了红蓝双方的初始位置和速度。
    # def get_init_pos(self):
    #     max_range = 0.3
    #     red_y = 2 * max_range * np.random.random() - max_range
    #     blue_y = 2 * max_range * np.random.random() - max_range
    #     initial_pos_set = [[max_range, -max_range, -90, 90],
    #                        [-max_range, max_range, 90, -90]]
    #     initial_pos = random.choice(initial_pos_set)
    #     r1 = 0.2 * np.random.random() + 0.8
    #     r2 = 0.2 * np.random.random() + 0.8
    #     r3 = 0.5 * np.random.random() + 0.5
    #     r4 = 0.5 * np.random.random() + 0.5
    #     red_x, blue_x, red_psi, blue_psi = \
    #         r1 * initial_pos[0], \
    #         r2 * initial_pos[1], \
    #         initial_pos[2], \
    #         initial_pos[3]
    #     red_v, blue_v = 900 * r3, 900 * r4
    #     return red_x, red_y, red_psi, red_v, blue_x, blue_y, blue_psi, blue_v

    def get_init_pos(self):
        max_range = 0.6
        red_y = 0.5 * np.random.random() - 0.25
        blue_y = 0.5 * np.random.random() - 0.25
        initial_pos_set = {
            'equal': [[-max_range, max_range, 90, -90]],
        }
        # 可以设置权重
        key_ = random.choice(list(initial_pos_set.keys()))
        initial_pos = random.choice(initial_pos_set[key_])
        r1 = 0.25 * np.random.random() + 0.75
        r2 = 0.25 * np.random.random() + 0.75
        r3 = 0.25 * np.random.random() + 0.75
        r4 = 0.25 * np.random.random() + 0.75
        r5 = 0.25 * np.random.random() + 0.75
        red_x, blue_x, red_psi, blue_psi = \
            r1 * initial_pos[0], \
            r2 * initial_pos[1], \
            initial_pos[2], \
            initial_pos[3]
        red_v, blue_v = 600 * r3, 600 * r4
        h = 32000 * r5
        return red_x, red_y, red_psi, red_v, blue_x, blue_y, blue_psi, blue_v, h


    def get_reward(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can calculate reward according to the obs

        Returns:
            calculated reward
        """
        reward = self.get_win_loss_reward()
        return reward

    def get_win_loss_reward(self):
        """
        Returns:
            win_loss reward
        """
        ############################################################
        # 首先考虑幕回报，当这一幕结束时给予的回报
        ############################################################
        control_side = self.control_side
        opponent_side = self.opponent_side
        if self.is_done['__all__']:
            control_side_death_event = self.obs_tot[control_side][f'{control_side}_0']['DeathEvent']
            opponent_side_death_event = self.obs_tot[opponent_side][f'{opponent_side}_0']['DeathEvent']
            # 1. 没打死
            if control_side_death_event == 99 and opponent_side_death_event == 99:
                return 0
            # 2. 红蓝都死了
            if control_side_death_event != 99 and opponent_side_death_event != 99:
                return 0
            # 3. 红方死了，蓝方没死
            if control_side_death_event != 99:
                return -1
            # 4. 红方没死，蓝方死了
            else:
                return 1
        return 0

    def judge_done(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can judge whether is_done according to the obs

        Returns:
            is_done or not
        """
        control_side = self.control_side
        opponent_side = self.opponent_side
        done = {}
        life_control_side = self.obs_tot[control_side][f'{control_side}_0']['LifeCurrent']
        life_opponent_side = self.obs_tot[opponent_side][f'{opponent_side}_0']['LifeCurrent']
        IfPresenceHitting_control_side = self.obs_tot[control_side][f'{control_side}_0']['IfPresenceHitting']
        IfPresenceHitting_opponent_side = self.obs_tot[opponent_side][f'{opponent_side}_0']['IfPresenceHitting']
        if life_control_side == 0:
            done[control_side] = True
        else:
            if life_opponent_side == 0 and IfPresenceHitting_opponent_side == 0:
                done[control_side] = True
            else:
                done[control_side] = False
        if life_opponent_side == 0:
            done[opponent_side] = True
        else:
            if life_control_side == 0 and IfPresenceHitting_control_side == 0:
                done[opponent_side] = True
            else:
                done[opponent_side] = False
        if done[opponent_side] and done[control_side]:
            done['__all__'] = True
        else:
            done['__all__'] = False
        max_step_num = param.Max_step_num
        if self.step_num >= max_step_num:
            done['__all__'] = True
        return done

    def _common_attribute_state_process(self):
        """
        :return: 返回状态编码信息
        """
        control_side = self.control_side
        opponent_side = self.opponent_side
        alley = ['base_state']
        opponent = ['base_state']
        post_process_obs_control_side = []
        post_process_obs_opponent_side = []
        post_process_obs_control_side_dict = self._single_player_state_process(flag=control_side)
        
        # # TODO 在真正比赛时无法获得完整的对手信息，仅可使用可探测到的信息，此处仅为示范
        # post_process_obs_opponent_side_dict = self._single_player_state_process(flag=opponent_side)
        post_process_obs_opponent_side_dict = {}
        if 'TargetIntoView' in post_process_obs_control_side_dict and post_process_obs_control_side_dict['TargetIntoView'] != 0:
            post_process_obs_opponent_side_dict = self._single_player_state_process(flag=opponent_side)

        for key in alley:
            post_process_obs_control_side.extend(post_process_obs_control_side_dict[key])
            post_process_obs_opponent_side.extend(post_process_obs_opponent_side_dict[key])
        for key in opponent:
            post_process_obs_control_side.extend(post_process_obs_opponent_side_dict[key])
            post_process_obs_opponent_side.extend(post_process_obs_control_side_dict[key])
        [death_control_side, death_opponent_side] = self.death_event()
        post_process_obs_control_side.extend([death_control_side, death_opponent_side])
        post_process_obs_opponent_side.extend([death_opponent_side, death_control_side])
        return post_process_obs_control_side, post_process_obs_opponent_side

    def death_event(self):
        control_side = self.control_side
        opponent_side = self.opponent_side
        post_process_obs = []
        control_side_death = self.obs_tot[control_side][f'{control_side}_0']['DeathEvent']
        if control_side_death == 99:
            post_process_obs.append(2)
        elif control_side_death == 0:
            post_process_obs.append(0)
        else:
            post_process_obs.append(1)
        opponent_side_death = self.obs_tot[opponent_side][f'{opponent_side}_0']['DeathEvent']
        if opponent_side_death == 99:
            post_process_obs.append(2)
        elif opponent_side_death == 0:
            post_process_obs.append(0)
        else:
            post_process_obs.append(1)
        return post_process_obs

    def _single_player_state_process(self, flag):
        """
        :return: 返回编码的状态信息
        """
        post_process_obs = dict()
        post_process_obs['base_state'] = self._state_process(flag=flag)
        return post_process_obs

    def _state_process(self, flag):
        """
        TODO：本函数是状态编码信息，此处只是编码了生命值和高度信息
        TODO：编程开发者可以参考智空文档，自行定义智能体训练需要的观测信息
        """
        # control_side = self.control_side
        # opponent_side = self.opponent_side
        if self.is_done[flag]:
            return [0 for i in range(23)]
        else:
            post_process_obs = []
            obs = self.obs_tot[flag][f'{flag}_0']
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
            
            #新增视野的敌机编号和进入攻击范围的
            post_process_obs.append(obs['TargetIntoView'])
            post_process_obs.append(obs['TargetEnterAttackRange'])
        # elif flag == opponent_side:
        #     #高度
        #     post_process_obs.append(obs['position/h-sl-ft'])
        #     # 经纬度
        #     post_process_obs.append(obs['position/long-gc-deg'])
        #     post_process_obs.append(obs['position/lat-geod-deg'])
        #     # 速度
        #     post_process_obs.append(obs['velocities/u-fps'])
        #     post_process_obs.append(obs['velocities/v-fps'])
        #     post_process_obs.append(obs['velocities/w-fps'])
        #     # 角速度
        #     post_process_obs.append(obs['velocities/p-rad_sec'])
        #     post_process_obs.append(obs['velocities/q-rad_sec'])
        #     post_process_obs.append(obs['velocities/r-rad_sec'])

            return post_process_obs