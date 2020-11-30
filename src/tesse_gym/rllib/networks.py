###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    from ray.rllib.models.modelv2 import ModelV2
    from ray.rllib.models.torch.misc import SlimConv2d, SlimFC, normc_initializer
    from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.annotations import override
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "ray or pytorch is not installed. Please install tesse-gym "
        "with the [rllib] option"
    )


class RLLibNetworkBase:
    def __init__(
        self, cnn: nn.Module, cnn_shape_hwc: Tuple[int, int, int], pose_length: int
    ):
        """ Helpers for rllib networks.

        Args:
            cnn (nn.Module): CNN modules.
            cnn_shape_hwc (Tuple[int, int, int]): Expected
                shape of `cnn` input in (height x width x channel).
            pose_length (int): Expected length of input pose
                at end of flattened observation.
        """
        self.cnn = cnn
        self.cnn_shape_hwc = cnn_shape_hwc
        self.pose_length = pose_length

    def process_images_and_pose(self, input: torch.Tensor) -> torch.Tensor:
        """ Split pose and imgs, process images through cnn.

        Args:
            input (torch.Tensor): (B, T, F) tensor
                where F = (h*w*c + pose_length).

        Returns:
            torch.Tensor: CNN features concatenated with pose.
        """
        flat_imgs = input[..., : -self.pose_length]
        imgs = torch.reshape(flat_imgs, (-1,) + self.cnn_shape_hwc)
        imgs = imgs.permute(0, 3, 1, 2).float()  # hwc -> chw

        img_features = self.cnn(imgs)

        if self.pose_length is not None:
            poses = input[..., -self.pose_length :]
            poses = poses.reshape(-1, poses.shape[-1])
            features = torch.cat((img_features, poses), dim=-1)
        else:
            features = img_features

        return features


class NatureCNN(nn.Module):
    def __init__(self, cnn_shape: Tuple[int, int, int]):
        """ Nature CNN

        Args:
            cnn_shape (Tuple[int, int, int]): CHW of expected image.
        """
        nn.Module.__init__(self)

        self.cnn_shape = cnn_shape
        cnn_downsample_factor = 8

        activ = nn.ReLU
        self.base = nn.Sequential(
            nn.Conv2d(self.cnn_shape[0], 32, (8, 8), 4, padding=4),
            activ(),
            nn.Conv2d(32, 64, (4, 4), 2, padding=1),
            activ(),
            nn.Conv2d(64, 64, (3, 3), 1, padding=1),
            activ(),
        )
        n_outputs = (
            self.cnn_shape[1]
            // cnn_downsample_factor
            * self.cnn_shape[2]
            // cnn_downsample_factor
            * 64
        )

        self.linear = nn.Linear(n_outputs, 512)

        for layer in self.base:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                # layer.bias.data.fill_(0.0)

    def forward(self, input_imgs: Tensor) -> Tensor:
        """ Call model on input tensor.

        Args:
            input_imgs (Tensor): flatted tensor of images
                and pose of shape [Batch, Time, obs_size].
        
        Returns:
            Tensor: Output of shape [Batch*Time, out_feature_size].
        """
        conv_out = self.base(input_imgs)
        conv_flattened = conv_out.view(conv_out.shape[0], -1)
        return self.linear(conv_flattened)


class NatureCNNActorCritic(TorchModelV2, nn.Module, RLLibNetworkBase):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
        cnn_shape: Tuple[int, int, int],
        pose_length: int,
    ):
        """ Actor Critic Model with CNN feature extractor.

        Args:
            obs_space (gym.spaces.Space): Agent's observation space.
            action_space (gym.spaces.Space): Agent's action space.
            num_outputs (int): Lenght of model output vector. 
                Corresponds to `action_space`.
            model_config (ModelConfigDict): Rllib specific configuration.
            name (str): Model name (scope).
            cnn_shape (Tuple[int, int, int]): (channel, height, width) of 
                expected input.
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # cnn setup
        self.cnn_shape = cnn_shape
        self.cnn_shape_hwc = (cnn_shape[1], cnn_shape[2], cnn_shape[0])
        self.cnn = NatureCNN(cnn_shape)

        # linear / policy setup
        self.pose_length = pose_length
        linear_in = 512 if self.pose_length is None else 512 + self.pose_length
        self.linear = nn.Linear(linear_in, 512)
        self.value_branch = nn.Linear(512, 1)
        self._features = None

        RLLibNetworkBase.__init__(self, self.cnn, self.cnn_shape_hwc, pose_length)

    @override(TorchModelV2)
    def forward(
        self, input_dict: Dict[str, Tensor], state: List[Tensor], seq_len: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        """ Call model with given input and state.

        Args:
            input_dict (Dict[str, Tensor]): Dict of input tensors including
                "obs", "obs_flat", "prev_action", "prev_reward", "is_training",
                "eps_id", "agent_id", infos", and "t".
            state (List[Tensor]): List of state tensors size of that returned
                by `get_initial_state` + batch_dimension.
            seq_lens (Tensor): 1d tensor holding input sequence lengths.

        Returns:
            Tuple[Tensor, List[Tensor]]: The model output tensor
                of size [BATCH, num_outputs] and new RNN state.
        """
        linear_in = self.process_images_and_pose(input_dict["obs"])
        self._features = self.linear(linear_in)
        return self.policy(self._features), state

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self.value_branch(self._features), [-1])


class NatureCNNRNNActorCritic(TorchRNN, nn.Module, RLLibNetworkBase):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
        cnn_shape: Tuple[int, int, int],
        rnn_type: Optional[str] = "LSTM",
        hidden_size: Optional[int] = 256,
        pose_length: int = 5,
    ):
        """ Actor Critic Model with CNN feature extractor.

        Args:
            obs_space (gym.spaces.Space): Agent's observation space.
            action_space (gym.spaces.Space): Agent's action space.
            num_outputs (int): Lenght of model output vector. 
                Corresponds to `action_space`.
            model_config (ModelConfigDict): Rllib specific configuration.
            name (str): Model name (scope).
            cnn_shape (Tuple[int, int, int]): (channel, height, width) of 
                expected input.
            rnn_type (Optional[str]): RNN type of either [LSTM, GRU].
            hidden_size (Optional[int]): RNN hidden state size.
        """
        TorchRNN.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.cnn_shape_hwc = (cnn_shape[1], cnn_shape[2], cnn_shape[0])
        self.cnn = NatureCNN(cnn_shape)

        self.pose_length = pose_length
        rnn_input_size = 512 if self.pose_length is None else 512 + self.pose_length
        self.rnn_state_size = hidden_size
        self.rnn = getattr(nn, rnn_type)(
            input_size=rnn_input_size, hidden_size=self.rnn_state_size, batch_first=True
        )

        self.policy = nn.Linear(self.rnn_state_size, num_outputs)
        self.value_branch = nn.Linear(self.rnn_state_size, 1)
        self._features = None

        RLLibNetworkBase.__init__(self, self.cnn, self.cnn_shape_hwc, self.pose_length)

    @override(TorchRNN)
    def forward_rnn(
        self, input: Tensor, state: List[Tensor], seq_len: Tensor
    ) -> Tuple[Tensor, List[Tensor]]:
        """ Call model with given input and state.

        Args:
            input_dict (Tensor): Input of shape [Batch, Time, obs_size].
            state (List[Tensor]): List of state tensors size of that returned
                by `get_initial_state` + batch_dimension.
            seq_lens (Tensor): 1d tensor holding input sequence lengths.

        Returns:
            Tuple[Tensor, List[Tensor]]: The model output tensor
                of size [BATCH, num_outputs] and new RNN state.
        """
        base_features = self.process_images_and_pose(input)

        # the CNN combines batch and time indices for inference
        # this needs to be decoupled for RNN
        base_features_time_ranked = torch.reshape(
            base_features, [input.shape[0], input.shape[1], base_features.shape[-1]],
        )

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            base_features_time_ranked, seq_len, batch_first=True, enforce_sorted=False
        )

        self._features, state = self.rnn(packed_input, self.preprocess_rnn_state(state))
        self._features = torch.nn.utils.rnn.pad_packed_sequence(
            self._features, batch_first=True
        )[0]

        return self.policy(self._features), self.postprocess_rnn_state(state)

    def preprocess_rnn_state(self, state: List[Tensor]) -> List[Tensor]:
        """ Reshape state as required by RNN inference. """
        return (
            torch.unsqueeze(state[0], 0)
            if isinstance(self.rnn, nn.GRU)
            else [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )

    def postprocess_rnn_state(self, state: List[Tensor]) -> List[Tensor]:
        """ Reshape state returned by RNN as required by rllib. """
        return (
            [state.squeeze(0)]
            if isinstance(self.rnn, nn.GRU)
            else [state[0].squeeze(0), state[1].squeeze(0)]
        )

    @override(ModelV2)
    def get_initial_state(self) -> List[Tensor]:
        """ Get initial RNN state consisting of zero vectors. """
        if isinstance(self.rnn, nn.GRU):
            h = [self.cnn.linear.weight.new(1, self.rnn_state_size).zero_().squeeze(0)]
        elif isinstance(self.rnn, nn.LSTM):
            h = [
                self.cnn.linear.weight.new(1, self.rnn_state_size).zero_().squeeze(0),
                self.cnn.linear.weight.new(1, self.rnn_state_size).zero_().squeeze(0),
            ]
        else:
            raise ValueError(f"{self.rnn} not supported RNN type")

        return h

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self.value_branch(self._features), [-1])

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass
