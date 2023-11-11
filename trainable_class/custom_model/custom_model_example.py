import logging
import numpy as np
import gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
torch, nn = try_import_torch()
logger = logging.getLogger(__name__)


class Custom_Model_Example(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space, num_outputs: int,
        model_config: ModelConfigDict, name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        ###################################################
        # 策略的隐藏层
        ###################################################
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size, out_size=size,
                    initializer=normc_initializer(1.0), activation_fn=activation
                ))
            prev_layer_size = size
        if len(hiddens) > 0:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size, out_size=hiddens[-1],
                    initializer=normc_initializer(1.0), activation_fn=activation,
                ))
            prev_layer_size = hiddens[-1]
        self._logits = SlimFC(
            in_size=prev_layer_size, out_size=num_outputs,
            initializer=normc_initializer(0.01), activation_fn=None,
        )
        self._hidden_layers = nn.Sequential(*layers)
        ###################################################
        # 值函数的网络部分，包括了值函数的编码网络与最后的输出层
        ###################################################
        prev_vf_layer_size = int(np.product(obs_space.shape))
        vf_layers = []
        for size in hiddens:
            vf_layers.append(
                SlimFC(
                    in_size=prev_vf_layer_size, out_size=size,
                    activation_fn=activation, initializer=normc_initializer(1.0),
                ))
            prev_vf_layer_size = size
        self._value_branch_separate = nn.Sequential(*vf_layers)
        self._value_branch = SlimFC(
            in_size=prev_layer_size, out_size=1,
            initializer=normc_initializer(0.01), activation_fn=None,
        )
        self._features = None
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(
            self._value_branch_separate(self._last_flat_in)
        ).squeeze(1)
