import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimConv2d, SlimFC, normc_initializer
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class NatureCNN(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, cnn_shape
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
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

        self.policy = nn.Linear(512, num_outputs)
        self.value_branch = nn.Linear(512, 1)
        self._features = None

        for layer in self.base:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.fill_(0.0)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_len):
        input = input_dict["obs"]
        input = input.permute(0, 3, 1, 2).float()
        conv_out = self.base(input)

        # todo (ZR) check this reshapes to (TIME, BATCH, FEATURES)
        self._features = self.linear(conv_out.view(conv_out.shape[0], -1))

        return self.policy(self._features), state

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self.value_branch(self._features), [-1])


class NatureCNNLSTM(TorchRNN, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, cnn_shape
    ):
        TorchRNN.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.cnn_shape = (5, 120, 160) #) cnn_shape
        cnn_downsample_factor = 8
        n_outputs = (
            self.cnn_shape[1]
            // cnn_downsample_factor
            * self.cnn_shape[2]
            // cnn_downsample_factor
            * 64
        )

        activ = nn.ReLU
        self.base = nn.Sequential(
            nn.Conv2d(self.cnn_shape[0], 32, (8, 8), 4, padding=4),
            activ(),
            nn.Conv2d(32, 64, (4, 4), 2, padding=1),
            activ(),
            nn.Conv2d(64, 64, (3, 3), 1, padding=1),
            activ(),
        )

        self.linear = nn.Linear(n_outputs, 512)

        self.lstm_state_size = 256
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=self.lstm_state_size, batch_first=True
        )

        self.policy = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        self._features = None

        for layer in self.base:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

    @override(TorchRNN)
    def forward_rnn(self, input, state, seq_len):
        input = input  # TODO(ZR) add pose
        # decode as (h, w, c)
        cnn_input_shape = (self.cnn_shape[1], self.cnn_shape[2], self.cnn_shape[0])
        inputs = torch.reshape(input, (-1,) + cnn_input_shape)
        inputs = inputs.permute(0, 3, 1, 2).float()
        conv_out = self.base(inputs)

        # todo (ZR) check this reshapes to (BATCH, TIME, FEATURES)
        base_features = self.linear(conv_out.view(conv_out.shape[0], -1)).unsqueeze(0)

        base_features_time_ranked = torch.reshape(
            base_features, [input.shape[0], input.shape[1], base_features.shape[-1]],
        )
        # if len(state[0].shape) == 2:
        #     state[0] = state[0].unsqueeze(0)
        #     state[1] = state[1].unsqueeze(1)

        self._features, [h, c] = self.lstm(
            base_features_time_ranked,
            [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)],
        )
        self._features = self._features

        return self.policy(self._features), [h.squeeze(0), c.squeeze(0)]

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self.value_branch(self._features), [-1])

    def get_weights(self):
        return None 

    def set_weights(self, weights):
        pass
