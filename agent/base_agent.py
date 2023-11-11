class Base_agent:
    def __init__(self, control_num, side, agent_config):
        """
        :param control_num: 我方需要控制的智能体数量
        :param side: 我方控制的阵营，包含'red'与'blue'
        """
        self.side = side
        self.control_num = control_num
        self.agent_config = agent_config
        self.obs_tot = None

    def reset(self):
        pass

    def agent_step(self, obs):
        """
        :param obs: 平台提供的返回信息
        :return: 需要向平台发送的操控信息
        """
        pass

