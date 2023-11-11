import socket
import json
import io
import zipfile
import os
import time
import gym


class Base_env(gym.Env):
    def __init__(self, config=None, render=0):
        """
        :param config: 从RLlib中传输过来的参数，在这个config里面可以传递希望定制的环境变量，譬如ip，render等
        :param render: 是否可视化
        """
        config_keys = config.keys()
        self.step_method_opponent = config['step_method_opponent'] \
            if 'step_method_opponent' in config_keys else 'time_triggered'
        self.frame_feature_size = config['frame_feature_size'] if 'frame_feature_size' in config_keys else 55
        self.control_side = config['control_side'] if 'control_side' in config_keys else 'red'
        self.opponent_side = 'blue' if self.control_side == 'red' else 'red'
        self.mode = config['mode'] if 'mode' in config_keys else 'train'
        self.usage = config['usage'] if 'usage' in config_keys else 'collect'
        self.IP = config['ip'] if 'ip' in config_keys else '127.0.0.1'
        self.PORT = (config['port'] if 'port' in config_keys else 8000) + config.worker_index
        self.RENDER = int(config['render']) if 'render' in config_keys else int(0)
        self.red_num = config['red_num'] if 'red_num' in config_keys else 1
        self.blue_num = config['blue_num'] if 'blue_num' in config_keys else 1
        self.red_com = config['Red'] if 'Red' in config_keys else 0
        self.blue_com = config['Blue'] if 'Blue' in config_keys else 0
        self.scenes = config['scenes'] if 'scenes' in config_keys else 3
        self.excute_path = config['excute_path'] if 'excute_path' in config_keys else '/home/user/linux/ZK.x86_64'
        self.data = None  # set for debug
        self.INITIAL = False
        self.excute_cmd = f'{self.excute_path} Ip={self.IP} Port={self.PORT} ' \
                          f'PlayMode={self.RENDER} ' \
                          f'RedNum={self.red_num} BlueNum={self.blue_num} ' \
                          f'Red={self.red_com} Blue={self.blue_com} Scenes={self.scenes}'
        self.create_entity()
        self.set_s_a_space()
        self.obs_tot = None
        self.is_done = {'__all__': False, 'red': False, 'blue': False}
        self.step_num = 0

    def create_entity(self):
        is_success = False
        while not is_success:
            try:
                self.excute_cmd = f'{self.excute_path} Ip={self.IP} Port={self.PORT} ' \
                                  f'PlayMode={self.RENDER} ' \
                                  f'RedNum={self.red_num} BlueNum={self.blue_num} ' \
                                  f'Red={self.red_com} Blue={self.blue_com} Scenes={self.scenes}'
                print('Creating Env', self.excute_cmd)
                self.unity = os.popen(self.excute_cmd)
                time.sleep(20)
                self._connect()
                is_success = True
                print('Env Created')
            except Exception as e:
                print('Create failed and the reason is ', e)
                time.sleep(5)

    def _connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(50)
        print(f'Connecting {self.IP}:{self.PORT}')
        self.socket.connect((self.IP, self.PORT))

    def reconstruct(self):
        print('Reconstruct Env')
        self.create_entity()
        self._connect()
        self.INITIAL = False

    def kill_env(self):
        print('Kill Env')
        output = os.popen(f'netstat -ano | findstr {self.IP}:{self.PORT}')
        output = output.read()
        output = output.split("\n")
        pid = None
        for out_tmp in output:
            out = out_tmp.split(' ')
            out_msg = []
            for msg_tmp in out:
                if msg_tmp != '':
                    out_msg.append(msg_tmp)
            try:
                if out_msg[1] == f'{self.IP}:{self.PORT}':
                    pid = out_msg[-1]
                    break
            except Exception as e:
                print('out_msg', out_msg)
        if pid is not None:
            os.system('taskkill /f /im %s' % pid)
            os.system('kill -9 %s' % pid)
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        self.INITIAL = False

    def _send_condition(self, data):
        self.socket.send(bytes(data.encode('utf-8')))
        self.data = data

    def _accept_from_socket(self):
        msg_receive = None
        load_data = None
        try:
            load_data = self.socket.recv(8192 * 16)
            zip_data = io.BytesIO(load_data)
            zip_file = zipfile.ZipFile(zip_data)
            msg_receive = zip_file.read(zip_file.namelist()[0])
            msg_receive = json.loads(str(msg_receive, encoding='utf-8'))
        except Exception as e:
            if e == socket.timeout:
                print('out of time')
            print("fail to recieve message from unity")
            print('load_data', load_data)
            print("the last sent data is: ", self.data)
            print(e)
            self._send_condition(self.data)
            load_data = self.socket.recv(8192 * 16)
            zip_data = io.BytesIO(load_data)
            zip_file = zipfile.ZipFile(zip_data)
            msg_receive = zip_file.read(zip_file.namelist()[0])
            msg_receive = json.loads(str(msg_receive, encoding='utf-8'))
        return msg_receive

    def get_obs(self):
        ask_info = {'flag': 'obs'}
        data = json.dumps(ask_info)
        self._send_condition(data)
        msg_receive = self._accept_from_socket()
        return msg_receive

    def get_obs_red(self):
        global_msg = self.get_obs()
        red_msg = global_msg['red']
        return red_msg

    def get_obs_blue(self):
        global_msg = self.get_obs()
        blue_msg = global_msg['blue']
        return blue_msg

    def reset(self, red_number: int = 1, blue_number: int = 1,
              reset_attribute: dict = None):
        self.reset_var()
        red_x, red_y, red_psi, red_v, blue_x, blue_y, blue_psi, blue_v, h = self.get_init_pos()
        # reset_attribute = {
        #     'red': {
        #         'red_0': {
        #             "ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08,
        #             "ic/long-gc-deg": red_x, "ic/lat-geod-deg": red_y,
        #             "ic/u-fps": red_v, "ic/v-fps": 0, "ic/w-fps": 0,
        #             "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
        #             "ic/roc-fpm": 0, "ic/psi-true-deg": red_psi}
        #              },
        #     'blue': {
        #         'blue_0': {
        #             "ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08,
        #             "ic/long-gc-deg": blue_x, "ic/lat-geod-deg": blue_y,
        #             "ic/u-fps": blue_v, "ic/v-fps": 0, "ic/w-fps": 0,
        #             "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
        #             "ic/roc-fpm": 0, "ic/psi-true-deg": blue_psi}
        #              }}
        reset_attribute = {
        'red': {
            'red_0': {
                "ic/h-sl-ft": h, "ic/terrain-elevation-ft": 1e-08,
                "ic/long-gc-deg": red_x, "ic/lat-geod-deg": red_y,
                "ic/u-fps": red_v, "ic/v-fps": 0, "ic/w-fps": 0,
                "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                "ic/roc-fpm": 0, "ic/psi-true-deg": red_psi}
                    },
        'blue': {
            'blue_0': {
                "ic/h-sl-ft": h, "ic/terrain-elevation-ft": 1e-08,
                "ic/long-gc-deg": blue_x, "ic/lat-geod-deg": blue_y,
                "ic/u-fps": blue_v, "ic/v-fps": 0, "ic/w-fps": 0,
                "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                "ic/roc-fpm": 0, "ic/psi-true-deg": blue_psi}
                    }}
        
        init_info = {'red': reset_attribute['red'],
                     'blue': reset_attribute['blue']}
        if self.INITIAL is False:
            self.INITIAL = True
            init_info['flag'] = {'init': {'render': self.RENDER}}
        else:
            init_info['flag'] = {'reset': {'render': self.RENDER}}
     
        data = json.dumps(init_info)
        self.is_done = {'__all__': False, 'red': False, 'blue': False}
        self.step_num = 0
        self._send_condition(data)
        obs_tot = self._accept_from_socket()
        self.obs_tot = obs_tot
        obs = self.postprocess_obs(self.obs_tot)
        return obs

    def step(self, action_attribute):
        action_attribute = self.postprocess_action(action_attribute)
        data = json.dumps(action_attribute)
        self._send_condition(data)
        obs_tot = self._accept_from_socket()
        info = {}
        self.obs_tot = obs_tot
        self.is_done = self.judge_done(self.obs_tot)
        obs = self.postprocess_obs(self.obs_tot)
        reward = self.get_reward(self.obs_tot)
        self.step_num += 1
        return obs, reward, self.is_done['__all__'], info

    def reset_var(self):
        pass

    # TODO: Need to be overwritten according to different work
    def postprocess_obs(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can normalize obs or postprocess obs here

        Returns:
            postprocessed obs
            {'red_0': obs_red,
            'blue_0': obs_blue}
        """
        raise NotImplementedError

    def get_init_pos(self):
        raise NotImplementedError

    # TODO: Need to be overwritten according to different work
    def get_reward(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can calculate reward according to the obs

        Returns:
            calculated reward
            {'red_0': reward1, 'blue_0': reward2}
        """
        raise NotImplementedError

    # TODO: Need to be overwritten according to different work
    def judge_done(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can judge whether is_done according to the obs

        Returns:
            is_done or not
            {
                "red": False,    # car_0 is still running
                "blue": True,     # car_1 is done
                "__all__": False,  # the env is not done
            }
        """
        raise NotImplementedError

    def postprocess_action(self, action):
        """
        Args:
            action: dict format, you can postprocess action according to different tasks
            {"red_0": [1,2,3,4],
            "blue_0": [2,3,4,5]})
        Returns:
            dict format action
            the same as action_input_example
        """
        raise NotImplementedError

    def set_s_a_space(self):
        """
        define state space and action space
        """
        raise NotImplementedError




