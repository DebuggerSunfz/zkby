"""
Author: Chenxu Qian
Email: qianchenxu@mail.nankai.edu.cn
使用RLlib库与强化学习算法训练1v1智能体的简单案例
"""

import param
from typing import Any, Dict
import ray
from ray.rllib.algorithms import ppo
from trainable_class.custom_policy.custom_policy_example import Custom_Trainable_Policy
from ray.tune.logger import pretty_print
from custom_env.DogFight import DogFight
import os
import atexit


@atexit.register
def exit():
    os.system("ps -ef|grep ZK.x86_64|grep -v grep |awk '{print $2}'|xargs kill -9")
    os.system("taskkill /F /IM ZK.exe")


def train(args, train_config: Dict[str, Any], train_num: int, save_folder=None, checkpoint_path=None) -> None:
    """Train the trainable_class with n_iters iterations."""

    agent = Custom_Trainable_Policy(config=train_config, env=train_config["env"])
    if checkpoint_path:
        agent.restore(checkpoint_path)
    while True:
        result = agent.train()
        train_num += 1
        print(pretty_print(result))
        if train_num % args.evaluation_interval == 0:
            try:
                reward_mean = result['evaluation']['episode_reward_mean']
            except Exception as e:
                reward_mean = 0
            checkpoint_path = agent.save(os.path.join(
                save_folder, 'checkpoint_%06d_%.1f' % (train_num, reward_mean)))
            print(f"Checkpoint saved in {checkpoint_path}")

if __name__ == "__main__":
    args = param.parser.parse_args()
    args.excute_path = r'/home/sunfz23/competition/MM/nolinux/ZK.x86_64'
    args.save_folder = '/home/sunfz23/competition/experiment/model_save_folder/tmp/'
    args.num_workers = 6
    args.evaluation_num_workers = 6
    args.checkpoint = None
    args.render = 0
    args.lr = 1e-5
    args.frame_feature_size = 48
    ray.init()

    config = {
        "env": DogFight,
        "env_config": {
            "render": args.render,
            'ip': '127.0.1.1',
            'port': args.port + 100,
            'Red': 0,
            'Blue': 0,  # Red和Blue用于通信，不同通信方式见文档
            'mode': 'train',
            'usage': 'train',
            "excute_path": args.excute_path,
            "frame_feature_size": args.frame_feature_size,
            'control_side': 'red',
        },
        'recreate_failed_workers': True,
        "num_gpus": args.num_gpus,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "use_gae": True,
        "lambda": 0.99,
        'model': {'fcnet_activation': 'tanh',
                  'custom_model_config': {}},

        'evaluation_num_workers': args.evaluation_num_workers,
        'evaluation_interval': args.evaluation_interval,
        'evaluation_config': {
            'exploration': False,
            "env_config": {
                "render": args.render,
                'ip': '127.0.1.1',
                'port': args.port,
                'mode': 'eval',
                'usage': 'train',
                "excute_path": args.excute_path,
                "frame_feature_size": args.frame_feature_size,
                'control_side': 'red',
            },
        },
        'evaluation_duration': args.evaluation_duration,
        "evaluation_duration_unit": "episodes",

        'seed': args.randomseed,
        "batch_mode": "complete_episodes",
        "train_batch_size": 1000,  # 每次训练的数据批，数据批越大，差距越小
        'sgd_minibatch_size': 500,  # 每次mini-batch-size的大小，这个值越大，计算越省时，主要也得看gpu是否使用，如果gpu不用，那大了也没意义
        "framework": "torch",
    }
    train_config = ppo.DEFAULT_CONFIG.copy()
    train_config.update(config)
    print("Start training.")
    train(args, train_config, train_num=args.train_num,
          save_folder=args.save_folder,
          checkpoint_path=args.checkpoint)
