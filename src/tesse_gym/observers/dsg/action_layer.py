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

from typing import List, Optional, Tuple, Union

import numpy as np
from tesse_gym.observers.dsg.dsg import NodeType

from tesse_gym.observers.dsg.esdf import SceneESDFHandler

NODE_TYPE_IDX = 6


class ActionLayer:
    def __init__(
        self,
        frame_type: str,
        params: Union[List[Tuple[int, float, float]], List[Tuple[float, int]]],
        edge_filter_ref: str,
        no_duplicate_layer_nodes: Optional[bool] = False,
        direct_esdf_lookup_node: Optional[bool] = False,
        direct_esdf_lookup_edge: Optional[bool] = False,
    ):
        """Agent frame layer.

        Places a fixed set of nodes around the agent.

        Args:
            frame_type (str): "node_frame" or "sector_frame".
            params (Union[List[Tuple[int, float, float]],
                          List[Tuple[float, int]]])

        Notes:
            "node_frame" takes params
                [nodes per layer, [dists], [edge thresholds]]
            "sector_frame" takes
                [(node distances, node angles)]
        """
        assert frame_type in ("node_frame", "sector_frame")
        assert edge_filter_ref in ("agent", "layer_node")

        self.type = frame_type
        self.__edge_filter_ref = edge_filter_ref

        if self.type == "node_frame":
            self.n_radials, self.radial_dists, self.edge_thresh = params
            self.new_nodes = self.n_radials * len(self.radial_dists)
        elif self.type == "sector_frame":
            self.config = params
            self.new_nodes = sum([angle for _, angle in self.config])
        else:
            raise ValueError(f"{self.type} not a recognized option")

        self.__no_duplcate_layer_nodes = no_duplicate_layer_nodes
        self.__direct_esdf_lookup_node = direct_esdf_lookup_node
        self.__direct_esdf_lookup_edge = direct_esdf_lookup_edge

    def _add_action_node(
        self,
        radial_layout,
        idx,
        dist,
        angle,
        esdf_handler,
        new_edges,
        nodes_to_consider,
        edge_thresh,
        new_nodes,
        nodes_to_consider_idx,
    ):
        # left-handed coords
        radial_layout[idx, 0] = dist * np.cos(-angle)
        radial_layout[idx, 1] = dist * np.sin(-angle)

        if esdf_handler is not None:
            agent_node_free = esdf_handler.is_valid_node(
                radial_layout[idx, :2],
                direct_esdf_lookup=self.__direct_esdf_lookup_node,
            )
        else:
            agent_node_free = True

        # add edges
        if agent_node_free:
            new_edges.extend(
                self._add_layer_node_edges(
                    radial_layout,
                    idx=idx,
                    nodes=nodes_to_consider,
                    edge_thresh=edge_thresh,
                    esdf_handler=esdf_handler,
                    new_nodes=new_nodes,
                    node_idxs=nodes_to_consider_idx,
                )
            )

        elif not agent_node_free:
            radial_layout[idx, 7] = 0  # TODO occupied

    def _add_node_frames(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        n_radials: int,
        radial_dists: List[float],
        edge_thresh: Union[float, List[float]],
        agent_frame_idx: Optional[int] = 5,
        esdf_handler: Optional[SceneESDFHandler] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add `node frame` to graph"""
        if not isinstance(edge_thresh, list):
            edge_thresh = [edge_thresh] * len(radial_dists)

        new_nodes = n_radials * len(radial_dists)
        radial_layout = np.zeros((new_nodes, nodes.shape[1]))
        radial_layout[:, 6:8] = agent_frame_idx
        new_edges = []

        radial_layout = self._add_esdf_value_if_needed(radial_layout)

        for i, angle in enumerate(
            np.linspace(
                0, 2 * np.pi, n_radials, endpoint=not self.__no_duplcate_layer_nodes
            )
        ):
            for j, dist in enumerate(radial_dists):
                idx = i + j * n_radials
                self._add_action_node(
                    radial_layout=radial_layout,
                    idx=idx,
                    dist=dist,
                    angle=angle,
                    esdf_handler=esdf_handler,
                    new_edges=new_edges,
                    nodes_to_consider=nodes,
                    edge_thresh=edge_thresh[j],
                    new_nodes=new_nodes,
                    nodes_to_consider_idx=np.arange(len(nodes)),
                )

        nodes, edges = self._extend_graph(nodes, edges, radial_layout, new_edges)

        return nodes, edges

    def _add_layer_node_edges(
        self,
        layer_nodes: np.ndarray,
        idx: int,
        nodes: np.ndarray,
        edge_thresh: float,
        esdf_handler: SceneESDFHandler,
        new_nodes: int,
        node_idxs: np.ndarray,
    ) -> List[Tuple[List[int], List[int]]]:
        new_edges = []
        dists = np.linalg.norm(
            layer_nodes[idx, (0, 1)] - nodes[np.newaxis, :, (0, 1)],
            axis=-1,
            ord=2,
        )
        layer_edges = np.where(dists < edge_thresh)[1]

        # raycast from layer node or agent
        if self.__edge_filter_ref == "layer_node":
            edge_filter_ref = layer_nodes[idx, :2]
        else:
            edge_filter_ref = np.zeros(2)
        valid_node_idx = self._filter_edges_via_esdf(
            esdf_handler,
            idx,
            edge_filter_ref,
            nodes,
            layer_edges,
            new_nodes,
            node_idx=node_idxs,
        )
        new_edges.extend(valid_node_idx)

        return new_edges

    def _get_node_of_type(self, nodes: np.ndarray, type: NodeType) -> np.ndarray:
        return np.where(nodes[:, NODE_TYPE_IDX] == type.value)[0]

    def _filter_edges_via_esdf(
        self,
        esdf_handler: SceneESDFHandler,
        idx: int,
        ref_pt,
        nodes,
        layer_edges,
        new_nodes,
        node_idx=None,
    ) -> Tuple[List[int], List[int]]:  # TODO(ZR) fix type
        new_edges = []
        if esdf_handler is not None:
            freespace_edges = esdf_handler.get_valid_edges_batch(
                ref_pt,
                nodes,
                layer_edges,
                direct_esdf_lookup=self.__direct_esdf_lookup_edge,
            )
        else:
            freespace_edges = layer_edges

        if node_idx is not None:
            freespace_edges = node_idx[freespace_edges]

        if len(freespace_edges):
            new_edges.append(
                (
                    idx * np.ones(len(freespace_edges)),
                    freespace_edges + new_nodes,
                )
            )
        return new_edges

    def _add_sector_frame(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        config: List[Tuple[float, int]],
        agent_frame_idx: Optional[int] = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add `sector frame` to graph"""
        node_dists, node_angles = self.cartesian_to_polar_coords(nodes[:, :2])
        # node_angles += np.pi  # [-pi, pi] -> [0, 2*pi]

        new_nodes = sum([a for _, a in config])
        frame = np.zeros((new_nodes, nodes.shape[1]))
        frame[:, 6:8] = agent_frame_idx
        new_edges = []

        frame = self._add_esdf_value_if_needed(frame)

        idx = 0
        prev_sector_dist = 0
        for (sector_dist, sector_angles) in config:
            angle_step = 2 * np.pi / sector_angles
            assert 360 % sector_angles == 0
            for angle in angle_step * np.arange(sector_angles):
                frame[idx, 0:2] = sector_dist * np.array(
                    [np.cos(-angle), np.sin(-angle)]
                )
                nodes_in_sector = self._get_nodes_in_sector(
                    node_dists,
                    node_angles,
                    sector_dist,
                    prev_sector_dist,
                    angle,
                    angle_step,
                )
                new_edges.append(
                    (idx * np.ones(nodes_in_sector.shape), nodes_in_sector + new_nodes)
                )
                idx += 1
            prev_sector_dist = sector_dist

        nodes, edges = self._extend_graph(nodes, edges, frame, new_edges)
        return nodes, edges

    def _add_esdf_value_if_needed(self, nodes: np.ndarray) -> np.ndarray:
        # if node features include ESDF (are length 10), assigne
        # frame nodes value -1
        # TODO(ZR) make conigurable
        if nodes.shape[1] == 10:
            nodes[:, 9] = -1
        return nodes

    def _get_nodes_in_sector(
        self,
        node_dists: np.ndarray,
        node_angles: np.ndarray,
        dist: float,
        prev_dist: float,
        angle: float,
        angle_step: float,
    ) -> np.ndarray:
        in_dist = np.logical_and(node_dists < dist, node_dists >= prev_dist)

        in_angles = node_angles - angle
        in_angles = self._bound_angles(in_angles)
        in_angles = in_angles < (angle_step / 2)

        return np.where(in_dist * in_angles)[0]

    @staticmethod
    def _extend_graph(
        nodes: np.ndarray,
        edges: np.ndarray,
        new_nodes: np.ndarray,
        new_edges: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(new_edges):
            new_edges = np.concatenate(new_edges, axis=-1).astype(np.uint32)
            new_edges = np.concatenate((new_edges, new_edges[::-1]), axis=1)
        nodes = np.concatenate((new_nodes, nodes), axis=0)
        edges += new_nodes.shape[0]
        if len(new_edges):
            edges = np.concatenate((edges, new_edges), axis=-1)
        return nodes, edges

    @staticmethod
    def cartesian_to_polar_coords(locations: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(locations, ord=2, axis=-1)
        # note counterclockwise rotation -> x/y
        theta = np.arctan2(locations[:, 0], locations[:, 1])
        return dist, theta

    @staticmethod
    def _bound_angles(x: np.ndarray) -> np.ndarray:
        """Bounds angles between [0, 2*pi]"""
        return np.abs((x + np.pi) % (2 * np.pi) - np.pi)

    def add_frame(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        agent_frame_idx: int,
        esdf_handler: Optional[SceneESDFHandler] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add the frame layer to the graph defined by `nodes`
        and `edges`

        Args:
            nodes (np.ndarray): (N, F) shape array of nodes, `N` is
                the number of nodes, `F` is features per node.
            edges (np.ndarray): (2, E) shape edge list. `E` is the
                number of edges.
            agent_frame_idx (int): Agent frame semantic value index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                tuple of nodes and edges.
        """
        if self.type == "node_frame":
            return self._add_node_frames(
                nodes,
                edges,
                self.n_radials,
                self.radial_dists,
                self.edge_thresh,
                agent_frame_idx,
                esdf_handler,
            )
        elif self.type == "sector_frame":
            return self._add_sector_frame(nodes, edges, self.config, agent_frame_idx)
