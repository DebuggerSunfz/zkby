import logging
from typing import Dict, List, Type, Union
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
)
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.utils.typing import TensorType
from trainable_class.custom_model.custom_model_example import Custom_Model_Example
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class Custom_Policy(
    PPOTorchPolicy
):
    """PyTorch policy class used with PPO.
    """

    def __init__(self, observation_space, action_space, config):  # 初始化Custom_Policy类，接受观察空间、动作空间和配置参数
        PPOTorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config
        )

    def make_model_and_action_dist(self):
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            self.action_space, self.config["model"], framework=self.framework
        )
        model = Custom_Model_Example(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=logit_dim,
            model_config=self.config["model"],
            name='Custom_Model_Example'
        )
        return model, dist_class

    @override(PPOTorchPolicy)
    def loss(
            self,
            model: ModelV2,
            dist_class: Type[ActionDistribution],
            train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.
        这个方法计算PPO算法的损失函数。它接受模型、动作分布类和训练批次数据作为输入，
        并返回损失张量。在此方法中，执行了许多PPO算法的核心计算，
        包括策略概率比率、KL散度、熵损失、值函数损失等。最后，所有损失被组合成一个总损失。
        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)    # 策略网络的输出logits和状态信息state
        curr_action_dist = dist_class(logits, model)  # 使用logits和模型model创建当前策略的动作分布curr_action_dist。这个分布描述了策略如何选择动作。

        # RNN case: Mask away 0-padded chunks at end of time axis.
        '''
        如果策略使用循环神经网络（RNN），则需要对序列进行处理
        在这种情况下，需要考虑序列的长度，以及如何处理序列中的0填充部分
        '''
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            # 创建一个掩码，用于标记有效的时间步
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)    #计算有效的时间步数量

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid
        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        # 获取之前的动作分布，这是根据采样动作计算的
        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )
        # 计算策略概率比率，即当前策略的动作选择概率与之前策略的概率之比。这是PPO算法的核心。
        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )
        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        # 计算策略损失，使用PPO算法中的surrogate loss公式
        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)   # 计算平均的策略损失

        # Compute a value function loss.  值函数损失
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = 0
            vf_loss_clipped = mean_vf_loss = 0.0

        # 综合所有损失项，包括策略损失、值函数损失、熵损失以及KL散度损失，权重由配置参数决定
        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # 存储损失值
        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss


class Custom_Trainable_Policy(PPO):
    def get_default_policy_class(self, config):
        return Custom_Policy
