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
from typing import List, Optional, Tuple

import numba
import numpy as np
import torch_geometric.utils as pyg_utils
from tesse_gym.observers.dsg.action_layer import ActionLayer
from tesse_gym.observers.dsg.dsg import (
    ESDFNode,
    NodeType,
    SceneGraph,
    load_graph_from_dict,
)
from tesse_gym.observers.dsg.esdf import SceneESDFHandler

try:
    import torch
    import torch_geometric.utils as pyg_utils
    from torch_geometric.data import Data
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "ray or pytorch is not installed. Please install tesse-gym "
        "with the [rllib] option"
    )


class DSGAccumulationModel:
    """Creates RL observations from Dynamic Scene Graph"""

    def __init__(
        self,
        scene_graph: str,
        graph_type: Optional[str] = "spark",
        esdf_handler: Optional[SceneESDFHandler] = None,
        accumulate_graph: Optional[bool] = False,
    ):
        """
        Args:
            scene_graph (str): Path to serialized scene graph.
            graph_type (str): Type of graph. Supported options are
                (`spark`, `spawn`).
            esdf_hander (Optiona[SceneESDFHanlder]): Handles ESDF logic.
            accumluate_graph (Optional[bool]): True to use
                iterative depth sensor model for graph accumulation.
                Otherwise, provide full graph.
        """
        # load serialized scene graph
        with open(scene_graph, "rb") as f:
            self._scene_graph = load_graph_from_dict(pickle.load(f, encoding="latin1"))

        # convert to pytorch geometric data structure
        if graph_type == "spark":
            converter = spark_scene_graph_to_pyg_data
        elif graph_type == "spawn":
            converter = scene_graph_to_pyg_data
        else:
            raise ValueError(
                f"Only graph types (`spark`, `spawn`)"
                f" are supported. Got {graph_type}kk"
            )
        self.__pyg_scene_graph = converter(self._scene_graph)
        self.__esdf_handler = esdf_handler

        if self.__esdf_handler is not None:
            places_points = self.__pyg_scene_graph.x
            self.__esdf_handler.set_esdf_unity_to_scene_graph_coords(
                places_points[:, :2].clone().numpy()
            )

        self.__seen_places_nodes = np.array([])
        self.__accumulate_graph = accumulate_graph

    def get_pyg_scene_graph(self) -> Data:
        return self.__pyg_scene_graph

    def get_esdf_handler(self) -> SceneESDFHandler:
        return self.__esdf_handler

    @staticmethod
    @numba.njit
    def remap_values(d_in: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        """For 2D array `d_in`, remap values in
        second axis to those in 1D array `mapping`

        Args:
            d_in (np.ndarray): Shape (M, N) array where
                second axis contains source values.
            mapping (np.ndarray): Shape (N, ) array
                of target values.
        Returns:
            np.ndarray: Shape (M, N) array containing
                values from `mapping`.

        """
        for i in range(len(mapping)):
            for j in range(d_in.shape[0]):
                for k in range(d_in.shape[1]):
                    if d_in[j, k] == mapping[i]:
                        d_in[j, k] = i
        return d_in

    @staticmethod
    @numba.njit
    def get_neighbors(node_idx: int, edges: np.ndarray) -> List[int]:
        """Get neighbors of `node_idx` from `edges`

        Args:
            node_idx (int): Query node index.
            edges (np.ndarray): Shape (2, N) array
                containing edges.
        """
        out = []
        for i in range(len(edges[0])):
            for j in range(len(node_idx)):
                if edges[0, i] == node_idx[j]:
                    out.append(edges[1, i])
        return out

    def reset(self) -> None:
        self.__seen_places_nodes = np.array([])

    def add_viewed_nodes(self, nodes: np.ndarray) -> None:
        """Add to set of viewed nodes

        Args:
            nodes (np.ndarray): Shape (N, ) array of
                node indices to add.
        """
        self.__seen_places_nodes = np.append(self.__seen_places_nodes, nodes)
        self.__seen_places_nodes = np.unique(self.__seen_places_nodes).astype(np.long)

    def get_nodes_in_radius(self, nodes: np.ndarray, dist: float) -> np.ndarray:
        """Get nodes within `dist` of origin.

        Args:
            nodes (np.ndarray): Shape (N, F) array of nodes where
                2D pose values are first two indices.
            dist (float): Distance threshold from origin.

        Returns:
            np.ndarray: Shape (M, F) array of nodes that are
                within `dist` of origin.
        """
        places_nodes_idx = np.where(nodes[:, NODE_TYPE_IDX] == NodeType.place.value)[0]
        places_nodes_idx = places_nodes_idx[
            np.linalg.norm(nodes[places_nodes_idx, :2], axis=-1, ord=2) < dist
        ]
        return places_nodes_idx

    def filter_by_angle(self, pts: np.ndarray, angle: float) -> np.ndarray:
        """Get indices of points in `pts` within `angle` degrees of
        origin.

        Args:
            pts (np.ndarray): Shape (N, 2) array of points.
            angle (float): Angle threshold in degrees.

        Returns
            np.ndarray: Indicies of points within `angle`
                degrees of origin.
        """
        _, angles = ActionLayer.cartesian_to_polar_coords(pts)

        angles = np.rad2deg(angles)

        return np.where(np.abs(angles) < angle / 2)[0]

    def get_nodes_via_accumulation_model(
        self, nodes: np.ndarray, esdf_handler: SceneESDFHandler, fov: Optional[int] = 90
    ) -> np.ndarray:
        """Use accumulation model to reveal DSG to agent.

        The accumulation model accounts for the agent's trajectory
        and sensor FOV to get portions of the DSG observed by the agent.

        Args:
            nodes (np.ndarray): (N, F) shape DSG node array
                where `N` is the number of nodes and `F` is
                feature length.
            esdf_hander (SceneESDFHander): Object to handle ESDF.
            fov (float): Camera field of view.

        Returns:
            np.ndarray: (N', F) array of observed DSG nodes where N` <= N
        """
        # get nodes within agent FOV
        _, angles = ActionLayer.cartesian_to_polar_coords(nodes[:, :2])
        angles = np.rad2deg(angles)
        nodes_in_fov_idx = np.where(np.abs(angles) < fov / 2)[0]

        # ignore nodes we've seen
        nodes_in_fov_idx = np.setdiff1d(nodes_in_fov_idx, self.__seen_places_nodes)

        # add places and object nodes
        node_types_in_fov = nodes[nodes_in_fov_idx, NODE_TYPE_IDX].astype(np.int32)
        places_nodes_in_fov_idx = nodes_in_fov_idx[
            np.logical_or(
                node_types_in_fov == NodeType.place.value,
                node_types_in_fov == NodeType.place.object,
            )
        ]
        places_nodes_in_fov = nodes[places_nodes_in_fov_idx]

        # can we see these nodes?
        start = np.zeros(2)
        closest_obstacle, _ = esdf_handler.bresenham_raycast(
            start, places_nodes_in_fov[:, :2]
        )
        closest_obstacle = np.array(closest_obstacle)
        obstacle_threshold = 0.05  # TODO(ZR) most objects will be occluded
        seen_node_idx = np.where(closest_obstacle > obstacle_threshold)[0]

        return places_nodes_in_fov_idx[seen_node_idx]

    def downsample_graph(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        method: Optional[str] = "dist",
        dist: Optional[float] = 2,
        esdf_handler: Optional[SceneESDFHandler] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get DSG nodes by distance to agent.

        Dowsample DSG using either accumulation model or distance
        from agent.

        Args:
            nodes (np.ndarray): Shape (N, F) array of DSG nodes
                where `N` is number of nodes and `F` is feature
                vector length.
            edges (np.ndarray): Shape (2, E) array of DSG edges.
            method (str in ["dist", "accumulation_mode"]):
                downsample method.
            dist (float): If using distance downsample,
                distance from agent to downsample around.
            esdf_hander (SceneESDFHander): Object to handle
                ESDF operations.
        """
        if method == "dist":
            places_nodes_idx = self.get_nodes_in_radius(nodes, dist)
        elif method == "accumulation_model":
            places_nodes_idx = self.get_nodes_via_accumulation_model(
                nodes, esdf_handler
            )
        else:
            raise ValueError(f"{method} not a recognized method")

        self.add_viewed_nodes(places_nodes_idx)

        # add viewed nodes neighbors (these will be room nodes)
        viewed_nodes = np.unique(
            np.append(
                self.__seen_places_nodes,
                self.get_neighbors(self.__seen_places_nodes, edges),
            )
        ).astype(np.long)

        # all edges connected to a visible place node
        places_edges = pyg_utils.subgraph(_tensor(viewed_nodes), _tensor(edges))[
            0
        ].numpy()

        # visible place nodes + connections
        visible_nodes_idx = np.unique(places_edges)
        visible_nodes = nodes[visible_nodes_idx]

        places_edges = self.remap_values(places_edges, visible_nodes_idx)

        return visible_nodes, places_edges

    def get_scene_graph(
        self,
        pose_translation: np.ndarray,
        pose_rotation: np.ndarray,
        node_pad_size: Optional[Tuple[int, int]] = None,
        edge_pad_size: Optional[Tuple[int, int]] = None,
        visited_points: Optional[np.ndarray] = None,
        polar_coords: Optional[bool] = False,
        agent_frame_handler: Optional[ActionLayer] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return graph nodes and edges.

        Args:
            pose_translation (np.ndarray): Shape (2,) translation
            pose_rotation (np.ndarray): Shape (3, 3) rotation
                matrix.
            node_pad_size (Optional[Tuple[int, int])): If given,
                zero pad node maxtrix to this size.
            edge_pad_size (Optional[Tuple[int, int]]): If given,
                zero pad node matrix to this size.
            visited_points (Optional[np.ndarray]): Shape (n, 3)
                trajectory vector of `n` poses in (x, z, yaw).
            polar_coords (Optional[bool]): True to provide
                graph node locations in polar coordinates relative
                to `pose_translation` and `pose_rotation`. Otherwise,
                cartesian coordinates are used. Default false.
            agent_frame_handler (Optional[AgentFrame]): Used to add
                agent frame to scene graph.
        """
        nodes = self.__pyg_scene_graph.x.clone().numpy()
        edges = self.__pyg_scene_graph.edge_index.clone().numpy()

        if visited_points is not None:
            self.mark_visited_points(nodes, visited_points)

        # convert points relative to spawn point
        xy_coords = (0, 1)
        nodes[:, xy_coords] -= pose_translation
        nodes[:, xy_coords] = np.matmul(pose_rotation, nodes[:, xy_coords].T).T

        if self.__esdf_handler is not None:
            self.__esdf_handler.transform_pts(pose_translation, pose_rotation)
            self.__esdf_handler.downsample_transformed_pts()

        if self.__accumulate_graph:
            nodes, edges = self.downsample_graph(
                nodes,
                edges,
                method="accumulation_model",
                esdf_handler=self.__esdf_handler,
            )

        if polar_coords:
            raise ValueError("Option currently not supported")

        if agent_frame_handler is not None:
            nodes, edges = agent_frame_handler.add_frame(
                nodes, edges, agent_frame_idx=5, esdf_handler=self.__esdf_handler
            )

        node_shape = np.array(nodes.shape)
        edge_shape = np.array(edges.shape)

        if node_pad_size is not None:
            nodes = self.pad_tensor(nodes, node_pad_size)

        if edge_pad_size is not None:
            edges = self.pad_tensor(edges, edge_pad_size)

        return nodes, edges, node_shape, edge_shape

    @staticmethod
    def pad_tensor(
        x: np.ndarray, target_shape: Tuple[int, ...], pad_val: Optional[float] = -1
    ) -> np.ndarray:
        """Right-pad tensor `x` to shape `target_shape` with `pad_val`"""
        pad_size = target_shape - np.array(x.shape)
        pad_size = [(0, s) for s in pad_size]
        return np.pad(x, pad_size, constant_values=pad_val)

    def mark_visited_points(
        self,
        node_data: np.ndarray,
        visited_points: np.ndarray,
        t: Optional[float] = 2,
        visited_coord: Optional[int] = 8,
    ) -> np.ndarray:
        """Mark nodes visited in the current episode with
            value of 1.

        Args:
            node_data (np.ndarray): Shape (n, f) matrix of `n` nodes
                with `f` features each.
            visited_points (np.ndarray): Shape (t, 3) matrix of
                `t` poses in (x, z, yaw).
            t (Optional[float]): Max distance from pose in
                trajectory for node to be considered visited.
            visited_coord (Optional[int]): Feature index to
                mark.

        Returns:
            np.ndarray: `node_data` with feature index `visited_coord`
                marked, if applicable.

        Notes:
            - Uses Unity coordinates, node centroid indices are (0, 2)
            - Uses node feature 8 to denote visitation

        """
        centroid_inds = (0, 1)
        type_coord = 6
        visited_points = torch.tensor(visited_points[:, :2])
        min_dists = np.expand_dims(node_data[:, centroid_inds], 1) - np.expand_dims(
            visited_points, 0
        )
        min_dists = np.linalg.norm(min_dists, ord=2, axis=-1)
        min_dists = min_dists.min(-1)
        node_data[:, visited_coord] = np.logical_and(
            min_dists < t, node_data[:, type_coord] == NodeType.place.value
        )


semantic_classes = {
    "Corner Stall": 0,
    "Hallway": 1,
    "Office": 2,
    "Restroom": 4,
    "building": 10,
    "place": 11,
    "Break Room": 20,
    "Storage Room": 21,
    "Cubical Space": 2,  # same as office
}

semantic_label_ind = 7
NODE_TYPE_IDX = 6
VISITED_IDX = 8


def spark_scene_graph_to_pyg_data(scene_graph: SceneGraph) -> Data:
    """Convert scene graph to pytorch geometric data type

    Args:
        scene_graph (SceneGraph): Scene graph object.

    Returns:
        Data: Pytorch geometric data type.
    """
    features = []
    pos = []
    for i in range(scene_graph.num_nodes()):
        if (scene_graph.get_node(i).size == 0).all():
            size = np.zeros(3)
        else:
            size = np.array(scene_graph.get_node(i).size)

        orig_semantic_label = scene_graph.get_node(i).semantic_label
        if orig_semantic_label in semantic_classes.keys():
            semantic_label = np.array([semantic_classes[orig_semantic_label]])
        else:
            assert (
                orig_semantic_label not in semantic_classes.values()
            ), f"Error: Reused semantic label {orig_semantic_label}"
            semantic_label = np.array([int(orig_semantic_label)])

        # xyz -> xzy
        centroid = scene_graph.get_node(i).centroid[..., (0, 2, 1)]
        visited = np.array([0])

        node_features = [  # length 9 vector
            centroid,  # 3
            size,  # 3
            np.array([scene_graph.get_node(i).node_type.value]),  # 1
            semantic_label,  # 1
            visited,  # 1
        ]

        if isinstance(scene_graph.get_node(i), ESDFNode):
            node_features.append(np.array([scene_graph.get_node(i).esdf_value]))

        features.append(np.concatenate(node_features))
        pos.append(centroid)

    features = np.array(features)
    pos = np.array(pos)
    pyg_edges = []

    adj_mtrx = scene_graph.generate_adjacency_matrix()
    pyg_edges = pyg_utils.dense_to_sparse(torch.tensor(adj_mtrx))[0]

    data = Data(
        x=_tensor(features).float(),
        edge_index=pyg_edges,
        pos=_tensor(pos),
    )
    return data


def _tensor(x: np.ndarray) -> torch.Tensor:
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
