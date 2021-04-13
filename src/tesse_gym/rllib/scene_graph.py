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

import pickle
from typing import Dict, Tuple

import numpy as np
from gym import spaces

from tesse.msgs import DataResponse
from tesse_gym.core.observations import ObservationConfig
from tesse_gym.core.tesse_gym import TesseGym

try:
    from torch_geometric.data import Data
    from scene_graph_learning.scene_graph import SceneGraph
    import torch
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "ray or pytorch is not installed. Please install tesse-gym "
        "with the [rllib] option"
    )


def get_scene_graph_task(BaseTask: TesseGym, **kwargs):
    """Add scene graph observations to a TesseGym task.

    This will dynamically inherit the task, `base_task` and
    append scene graphs to the observations

    Args:
        base_task (TesseGym): Task for which to add
            scene graph observations
        kwargs (Dict[str, Any]): Keyword args for `base_task`
            and SceneGraphTask.

    Returns:
        SceneGraphTask: `base_task` with a scene
            graph added to the observations.
    """

    class SceneGraphTask(BaseTask):
        def __init__(self, **kwargs):
            assert "scene_graph" in kwargs.keys()
            scene_graph = kwargs.pop("scene_graph")
            self.scene_graph_handlers = {}

            # get the scene graph for each scene
            for i, fp in scene_graph.items():
                self.scene_graph_handlers[int(i)] = SceneGraphHandler(fp)

            self.max_node_shape = np.array(
                [
                    sgh._pyg_scene_graph.x.numpy().shape
                    for sgh in self.scene_graph_handlers.values()
                ]
            ).max(0)
            self.max_edge_shape = np.array(
                [
                    sgh._pyg_scene_graph.edge_index.numpy().shape
                    for sgh in self.scene_graph_handlers.values()
                ]
            ).max(0)

            # Add graph to observation space
            obs_config = kwargs.pop("observation_config")
            obs_config = ObservationConfig(
                modalities=obs_config.modalities,
                pose=obs_config.pose,
                height=obs_config.height,
                width=obs_config.width,
                min=obs_config.min,
                max=obs_config.max,
                custom_obs={
                    "GRAPH_NODES": (-np.Inf, np.Inf, self.max_node_shape),
                    "GRAPH_EDGES": (-np.Inf, np.Inf, self.max_edge_shape),
                    "GRAPH_NODE_SHAPE": (-np.Inf, np.Inf, (2,)),
                    "GRAPH_EDGE_SHAPE": (-np.Inf, np.Inf, (2,)),
                },
            )
            kwargs["observation_config"] = obs_config

            super().__init__(**kwargs)

        def form_agent_observation(
            self, tesse_data: DataResponse
        ) -> Dict[str, np.ndarray]:
            """Add graph to observation.

            Args:
                tesse_data (DataResponse): Data read from TESSE.

            Returns:
                Dict[str, np.ndarray] Oservation.

            Notes:
                Nodes of shape (N, F), where `N` is the number of
                nodes and `F` is the node feature length, are added
                under the key `GRAPH_NODES`.

                Edges of shape (2, E), where `E` is the number of
                directional edges, are added under the key
                `GRAPH_EDGES`
            """
            observation = super().form_agent_observation(tesse_data)
            assert isinstance(observation, dict)
            nodes, edges, node_shape, edge_shape = self.scene_graph_handlers[
                self.current_scene
            ].get_scene_graph(
                node_pad_size=self.max_node_shape,
                edge_pad_size=self.max_edge_shape,
                pose_offset=self.initial_pose[:2],
            )
            observation["GRAPH_NODES"] = nodes
            observation["GRAPH_EDGES"] = edges
            observation["GRAPH_NODE_SHAPE"] = node_shape.astype(np.float32)
            observation["GRAPH_EDGE_SHAPE"] = edge_shape.astype(np.float32)
            return observation

    return SceneGraphTask(**kwargs)


class SceneGraphHandler:
    """ Creates RL observations from Dynamic Scene Graph """

    def __init__(self, scene_graph: str):
        """
        Args:
            scene_graph (str): Path to serialized scene graph.
        """
        # load serialized scene graph
        with open(scene_graph, "rb") as f:
            self._scene_graph = pickle.load(f)

        # convert to pytorch geometric data structure
        self._pyg_scene_graph = scene_graph_to_pyg_data(self._scene_graph)

    def get_scene_graph(
        self, node_pad_size: None, edge_pad_size: None, pose_offset=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Return graph nodes and edges. """
        nodes = self._pyg_scene_graph.x.clone().numpy()

        # make pose relative to agent spawn point
        if pose_offset is not None:
            nodes[:, 1:] -= pose_offset

        node_shape = np.array(nodes.shape)
        edges = self._pyg_scene_graph.edge_index.numpy()
        edge_shape = np.array(edges.shape)
        assert node_shape.all() and edge_shape.all(), "Empty graph!"

        if node_pad_size is not None:
            pad_size = node_pad_size - np.array(nodes.shape)
            pad_size = [(0, s) for s in pad_size]
            nodes = np.pad(nodes, pad_size, constant_values=-1)

        if edge_pad_size is not None:
            pad_size = edge_pad_size - np.array(edges.shape)
            pad_size = [(0, s) for s in pad_size]
            edges = np.pad(edges, pad_size, constant_values=-1)

        return nodes, edges, node_shape, edge_shape


def _tensor(x):
    return torch.tensor(x)


def scene_graph_to_pyg_data(scene_graph: SceneGraph) -> Data:
    """Convert scene graph to pytorch geometric data type

    Args:
        scene_graph (SceneGraph): Scene graph object.

    Returns:
        Data: Pytorch geometric data type.
    """
    x = np.arange(scene_graph.num_nodes())
    pos = np.array(
        np.array(
            [scene_graph.get_node(i).centroid for i in range(scene_graph.num_nodes())]
        )
    )
    pyg_edges = []

    adj_mtrx = scene_graph.generate_adjacency_matrix()
    for i in range(len(adj_mtrx)):
        for j in range(len(adj_mtrx)):
            if adj_mtrx[i, j]:
                pyg_edges.append([i, j])
    pyg_edges = np.array(pyg_edges).T

    data = Data(
        x=_tensor(np.concatenate([np.expand_dims(x, 1), pos], -1)).float(),
        edge_index=_tensor(pyg_edges),
        pos=_tensor(pos),
    )
    return data
