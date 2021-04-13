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
    from torch_geometric.nn import GCNConv
    import torch_geometric.nn as pyg_nn
    from ray.rllib.models.modelv2 import ModelV2
    from ray.rllib.models.torch.misc import SlimConv2d, SlimFC, normc_initializer
    from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.annotations import override
    from ray.rllib.models.modelv2 import restore_original_dimensions
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "ray or pytorch is not installed. Please install tesse-gym "
        "with the [rllib] option"
    )


class RLLibNetworkBase:
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        cnn: nn.Module,
        cnn_shape_hwc: Tuple[int, int, int],
        pose_length: int,
        gcn: nn.Module,
    ):
        """ Helpers for rllib network data preprocessing and inference.

        Args:
            obs_space (gym.spaces.Space): Expected observation space. RLLib
                flattens all spaces, the the original information is recovered 
                via `obs_space.original_space`.
            cnn (nn.Module): CNN modules.
            cnn_shape_hwc (Tuple[int, int, int]): Expected
                shape of `cnn` input in (height x width x channel).
            pose_length (int): Expected length of input pose
                at end of flattened observation.
            gcn (nn.Module): Graph Convolutional Network.
        """
        self.obs_space = obs_space.original_space
        self.full_obs_space = obs_space
        self.cnn = cnn
        self.cnn_shape_hwc = cnn_shape_hwc
        self.pose_length = pose_length
        self.gcn = gcn

    def process_images_and_pose(self, input: torch.Tensor) -> torch.Tensor:
        """ Split pose and imgs, process images through cnn.

        Args:
            input (torch.Tensor): (B, T, F) tensor
                where F = (h*w*c + pose_length).

        Returns:
            torch.Tensor: CNN features concatenated with pose.

        Notes:
            For Box observation spaces, images and poses
            will be extracted and properly formatted according to the 
            expected shapes given by `self.cnn_shape_hwc` and `self.pose_length`.
            
            If the observation space is a dictionary, image and vector types 
            will be inferred from the Space shape. Images will be stacked and 
            reshaped to (h,w,c).
        """
        # TODO(ZR) we need to specify which observations go where.
        # This is currently hardcoded, but should be configurable
        IMAGE_KEYS = ("RGB_LEFT", "SEGMENTATION", "DEPTH")
        POSE_KEYS = ("POSE",)
        if isinstance(self.obs_space, gym.spaces.Box):
            flat_imgs = input[..., : -self.pose_length]
            imgs = torch.reshape(flat_imgs, (-1,) + self.cnn_shape_hwc)
            poses = input[..., -self.pose_length :].reshape(-1, self.pose_length)
        elif isinstance(self.obs_space, gym.spaces.Dict):
            obs = restore_original_dimensions(
                input, self.full_obs_space, tensorlib="torch"
            )
            imgs = []
            graph_nodes = None
            graph_edges = None
            graph_node_shapes = None
            graph_edge_shapes = None
            for k, data in obs.items():
                # combine time and batch inds
                data = data.reshape((-1,) + tuple(data.shape[2:]))
                if k in IMAGE_KEYS:
                    imgs.append(data)
                elif k in POSE_KEYS:
                    poses = data
                elif k == "GRAPH_NODES":
                    graph_nodes = data
                elif k == "GRAPH_EDGES":
                    graph_edges = data
                elif k == "GRAPH_NODE_SHAPE":
                    graph_node_shapes = data.type(torch.int32)
                elif k == "GRAPH_EDGE_SHAPE":
                    graph_edge_shapes = data.type(torch.int32)
                else:
                    raise ValueError(f"Unexpected data shape: {data.shape}")

            imgs = torch.cat(imgs, dim=-1)

        imgs = imgs.permute(0, 3, 1, 2).float()  # hwc -> chw
        img_features = self.cnn(imgs)

        features = [img_features]
        if self.pose_length is not None:
            features.append(poses)
        if self.gcn is not None:
            # torch geometric doesn't support standard batch processing.
            # Rather, node and edge vectors are stacked and individual
            # graphs are denoted by a batch_index vector. We can get away
            # iteratively inferring now, but TODO(ZR) will extend this.
            # (Won't change results, just is cleaner and will speed this up).
            graph_features = []
            for i in range(graph_nodes.shape[0]):
                # If needed, rllib will 0 pad observations. This is fine for
                # image and vector processing, but 0 valued edge indices will break
                # a gcn model. Thus, if a graph is 0 padded (as denoted by the
                # expected shape), skip and add 0 to the resultant feature vector.
                if (graph_node_shapes[i] > 0).all():
                    gn = graph_nodes[
                        i, : graph_node_shapes[i][0], : graph_node_shapes[i][1]
                    ]
                    ge = graph_edges[
                        i, : graph_edge_shapes[i][0], : graph_edge_shapes[i][1]
                    ]
                    attention_features = (
                        poses[i].reshape(1, 3) if self.cat_pose_to_gcn else None
                    )
                    graph_features.append(
                        self.gcn(
                            gn,
                            ge.type(torch.int64),
                            final_concat_features=attention_features,
                        )
                    )
                else:
                    prev_features = graph_features[-1]
                    graph_features.append(
                        torch.zeros(prev_features.shape).to(prev_features.device)
                    )

            graph_features = torch.cat(graph_features, dim=0)
            features.append(graph_features)

        features = torch.cat(features, dim=-1)

        return features


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        graph_conv_features: List[int] = [16, 32],
        pooling_method="mean",
        final_concat_feature_length=0,
    ):
        """ Initialize Graph Convolutional Network

        ReLU is applied after each graph conv. 
        Global mean pooling is applied at the end.

        Args:
            in_features (int): Features per node.
            graph_conv_features (List[int]): Graph convolution
                features size, defines size of the network.
        """
        super(GCN, self).__init__()
        self.out_features = graph_conv_features[-1] + final_concat_feature_length
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(GCNConv(in_features, graph_conv_features[0]))
        self.pooling_method = pooling_method

        if "GlobalAttention" in pooling_method:
            self.attention = pyg_nn.GlobalAttention(
                nn.Linear(graph_conv_features[-1] + final_concat_feature_length, 1)
            )

        for i in range(1, len(graph_conv_features)):
            self.graph_convs.append(
                GCNConv(graph_conv_features[i - 1], graph_conv_features[i])
            )

    def forward(
        self, x: Tensor, edge_index: Tensor, final_concat_features=None
    ) -> Tensor:
        """ Pass graph, defined by `x` and `edge_index`, through network.

        Args:
            x (Tensor): Shape (N, F) tensor, `N` is the number of nodes
                `F` is the node feature length.
            edge_index (Tensor): Shape (2, E) tensor, `E` is the number 
                of directed edges.

        Notes:
            Graphs are stacked when performing batch inference.
            TODO(ZR) add option to pass in `batch_index`.
        """
        h = x
        for conv in self.graph_convs:
            h = conv(h.float(), edge_index).relu()

        if final_concat_features is not None:
            h = torch.cat((h, final_concat_features.repeat(h.shape[0], 1)), dim=-1)

        batch_index = torch.zeros(x.shape[0], dtype=torch.int64).to(x.device)
        if self.pooling_method == "mean":
            h = pyg_nn.global_mean_pool(h, batch_index)
        elif "GlobalAttention" in self.pooling_method:
            h = self.attention(h, batch_index)
        return h


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
        conv_flattened = conv_out.reshape(conv_out.shape[0], -1)
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
            obs_space (gym.spaces.Space): Agent's observation space. RLlib 
                flattens all spaces, so the original information is recovered
                via `obs_space.original_space`. This is used for preprocessing
                raw image, pose, etc. data.
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

        RLLibNetworkBase.__init__(
            self, obs_space, self.cnn, self.cnn_shape_hwc, pose_length, None
        )

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
        pose_length: Optional[int] = 5,
        graph_node_features: Optional[int] = None,
        graph_conv_features: Optional[List[int]] = [16, 32],
        graph_pooling_method: Optional[str] = "mean",
        cat_pose_to_gcn: Optional[bool] = False,
    ):
        """ Actor Critic Model with CNN feature extractor.

        Args:
            obs_space (gym.spaces.Space): Agent's observation space. RLlib 
                flattens all spaces, so the original information is recovered
                via `obs_space.original_space`. This is used for preprocessing
                raw image, pose, etc. data.
            action_space (gym.spaces.Space): Agent's action space.
            num_outputs (int): Lenght of model output vector. 
                Corresponds to `action_space`.
            model_config (ModelConfigDict): Rllib specific configuration.
            name (str): Model name (scope).
            cnn_shape (Tuple[int, int, int]): (channel, height, width) of 
                expected input.
            rnn_type (Optional[str]): RNN type of either [LSTM, GRU].
            hidden_size (Optional[int]): RNN hidden state size.
            pose_length (Optional[int]): Expected pose length.
            graph_node_features (Optional[int]): Input graph node features.
                If `None`, no graph is expected.
            graph_conv_features (Optional[List[int]]): Graph convolution
                features sizes. Defines size of network.
        """
        TorchRNN.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.cnn_shape_hwc = (cnn_shape[1], cnn_shape[2], cnn_shape[0])
        self.cnn = NatureCNN(cnn_shape)
        self.cat_pose_to_gcn = cat_pose_to_gcn
        if graph_node_features is not None:
            self.gcn = GCN(
                in_features=graph_node_features,
                graph_conv_features=graph_conv_features,
                pooling_method=graph_pooling_method,
                final_concat_feature_length=3 if cat_pose_to_gcn else 0,
            )
        else:
            self.gcn = None
        print(self.gcn)

        self.pose_length = pose_length
        rnn_input_size = 512
        if self.pose_length is not None:
            rnn_input_size += self.pose_length
        if graph_node_features is not None:
            rnn_input_size += self.gcn.out_features
        self.rnn_state_size = hidden_size
        self.rnn = getattr(nn, rnn_type)(
            input_size=rnn_input_size, hidden_size=self.rnn_state_size, batch_first=True
        )

        self.policy = nn.Linear(self.rnn_state_size, num_outputs)
        self.value_branch = nn.Linear(self.rnn_state_size, 1)
        self._features = None

        RLLibNetworkBase.__init__(
            self,
            obs_space,
            self.cnn,
            self.cnn_shape_hwc,
            self.pose_length,
            gcn=self.gcn,
        )

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
