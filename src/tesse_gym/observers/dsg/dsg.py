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

from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import List, Optional

import numpy as np

"""
Scene Graph, SceneNode and SceneEdge classes
"""


class SceneGraph:
    """
    SceneGraph class
      nodes      a list of SceneNode
      edge_dict  dictionary of SceneEdges {(start_idx, end_idx): a list of SceneEdge between start and end nodes} # TODO
    """

    def __init__(self):
        self.__nodes = []
        self.__edge_dict = dict()

    def num_nodes(self, node_type=None):
        if node_type is None:
            return len(self.__nodes)
        else:
            return sum(node.node_type == node_type for node in self.__nodes)

    def num_edges(self):
        return sum([len(v) for v in self.__edge_dict.values()])

    def get_node(self, node_idx):
        return self.__nodes[node_idx]

    def get_edge(self, start_idx, end_idx, rel):
        return next(
            (
                edge
                for edge in self.__edge_dict[(start_idx, end_idx)]
                if edge.rel == rel
            ),
            None,
        )

    def get_edge_relationships(self, start_idx, end_idx):
        return [edge.rel for edge in self.__edge_dict[(start_idx, end_idx)]]

    def get_edges(self, start_idx, end_idx):
        return self.__edge_dict[(start_idx, end_idx)]

    def get_nodes_copy(self):
        return deepcopy(self.__nodes)

    def get_edge_dict_copy(self):
        return deepcopy(self.__edge_dict)

    def get_adjacent_node_indices(self, node_idx):
        out_indices = [
            idx_pair[1]
            for idx_pair in list(self.__edge_dict.keys())
            if idx_pair[0] == node_idx
        ]
        in_indices = [
            idx_pair[0]
            for idx_pair in list(self.__edge_dict.keys())
            if idx_pair[1] == node_idx
        ]
        return out_indices, in_indices

    def set_node(self, node_idx, node):
        self.__nodes[node_idx] = node

    def find_parent_idx(self, scene_node):
        if scene_node.node_type == NodeType.building:
            return None

        node_idx = self.__nodes.index(scene_node)
        expected_type = (
            NodeType.building
            if scene_node.node_type == NodeType.room
            else NodeType.room
        )
        parent_indices = [
            idx_pair[1]
            for idx_pair in list(self.__edge_dict.keys())
            if idx_pair[0] == node_idx
            and self.__nodes[idx_pair[1]].node_type == expected_type
        ]
        if len(parent_indices) == 0:
            return None
        elif len(parent_indices) == 1:
            return parent_indices[0]
        else:
            print(
                "Warning: {} has more than one parent.".format(self.__nodes[node_idx])
            )
            return parent_indices[0]

    def get_relationship_set(self):
        return set(scene_edge.rel for scene_edge in sum(self.__edge_dict.values(), []))

    def add_node(self, new_node):
        # assert isinstance(new_node, SceneNode)
        if new_node not in self.__nodes:
            self.__nodes.append(new_node)

    def add_edge(self, new_edge):
        # assert isinstance(new_edge, SceneEdge)
        if new_edge.weight == 0:  # do not update when weight is 0
            return

        # update self.__nodes
        try:
            start_idx = self.__nodes.index(new_edge.start)
        except ValueError:
            start_idx = len(self.__nodes)
            self.__nodes.append(new_edge.start)  # make shallow copy
        try:
            end_idx = self.__nodes.index(new_edge.end)
        except ValueError:
            end_idx = len(self.__nodes)
            self.__nodes.append(new_edge.end)  # make shallow copy

        # update self.__edge_dict
        # TODO: delete print after debugging
        if (start_idx, end_idx) in self.__edge_dict.keys():
            try:
                edge_idx = self.__edge_dict[(start_idx, end_idx)].index(new_edge)
                self.__edge_dict[(start_idx, end_idx)][edge_idx] = new_edge
                print("Update weight of edge {}".format(new_edge))
            except ValueError:
                print(
                    "Additional relationship ({}) between scene node {} and {}".format(
                        new_edge.rel, new_edge.start, new_edge.end
                    )
                )
                self.__edge_dict[(start_idx, end_idx)].append(new_edge)
        else:
            self.__edge_dict[(start_idx, end_idx)] = [new_edge]

    def generate_adjacency_matrix(self):
        nr_nodes = len(self.__nodes)
        adjacency_matrix = np.zeros((nr_nodes, nr_nodes), dtype=bool)

        # A[i, j] = True when there is an edge from the i-th node to the j-th node in self.nodes
        start_indices = [edge_indices[0] for edge_indices in self.__edge_dict.keys()]
        end_indices = [edge_indices[1] for edge_indices in self.__edge_dict.keys()]
        adjacency_matrix[start_indices, end_indices] = True
        return adjacency_matrix


class SceneEdge:
    """
    SceneEdge class
      start      SceneNode
      rel        string or None for unknown relationship (ConceptNet and VG relationships or None)
      end        SceneNode
      weight     float
    """

    def __init__(self, start, rel, end, weight=1.0):
        # assert isinstance(start, SceneNode)
        # assert isinstance(end, SceneNode)
        self.start = start
        self.rel = rel
        self.end = end
        self.weight = weight

    def __str__(self):
        return "{0} - {1} - {2}".format(self.start, self.rel, self.end)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        # start, rel, end all have to be the same, but weight does not matter
        if not isinstance(other, SceneEdge):
            # don't attempt to compare against unrelated types
            return NotImplemented

        if not (
            self.start == other.start
            and self.rel == other.rel
            and self.end == other.end
        ):
            return False
        elif self.weight != other.weight:  # TODO: remove after debug
            print("same scene edge with different weight")
            return True
        else:
            return True


class NodeType(Enum):
    human = 0  # not used right now
    object = 1
    room = 2
    building = 3
    place = 4  # not used by CRF class


class SceneNode:
    """
    SceneNode class
      node_id             int (unique for each node in the same graph)
      node_type           SceneNodeType (objects, rooms, etc. or layer)
      semantic_label      string
      centroid            1d numpy array
      size                1d numpy array or None # TODO: on hold
      possible_labels     a list of strings or None
    """

    def __init__(
        self,
        node_id,
        node_type,
        centroid,
        size=None,
        semantic_label=None,
        possible_labels=None,
    ):
        # assert isinstance(node_type, NodeType)

        self.node_id = node_id
        self.node_type = node_type
        self.semantic_label = semantic_label
        self.centroid = np.array(centroid)
        self.size = None if size is None else np.array(size)
        self.possible_labels = possible_labels

    def __str__(self):
        semantic_label = (
            self.semantic_label if self.semantic_label is not None else "None"
        )
        return "%s (%d)" % (semantic_label, self.node_id)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.node_id, self.node_type, self.semantic_label))

    def __eq__(self, other):
        # compare id, node_type and semantic_label
        if (
            self.node_id == other.node_id
            and self.node_type == other.node_type
            and self.semantic_label == other.semantic_label
        ):
            return True
        elif (
            self.node_id == other.node_id and self.node_type == other.node_type
        ):  # Todo: for debugging
            print("Same node id and type but different semantic label")
            return False
        else:
            return False


class ESDFNode(SceneNode):
    def __init__(
        self,
        node_id: int,
        node_type: int,
        centroid: np.ndarray,
        size: np.ndarray = None,
        semantic_label: Optional[str] = None,
        possible_labels: Optional[List[str]] = None,
        esdf_value: Optional[float] = np.NaN,
    ):
        super().__init__(
            node_id,
            node_type,
            centroid,
            size=size,
            semantic_label=semantic_label,
            possible_labels=possible_labels,
        )
        self.esdf_value = esdf_value

    @staticmethod
    def from_node(
        scene_node: SceneNode, esdf_value: Optional[float] = None
    ) -> SceneNode:
        esdf_node = ESDFNode(
            scene_node.node_id,
            scene_node.node_type,
            scene_node.centroid,
            scene_node.size,
            scene_node.semantic_label,
            scene_node.possible_labels,
        )
        if esdf_value is not None:
            esdf_node.esdf_value = esdf_value
        return esdf_node


def encode_graph_as_dict(scene_graph):
    nodes = []

    for i in range(scene_graph.num_nodes()):
        n = scene_graph.get_node(i)
        node_dict = n.__dict__.copy()
        node_dict["node_type"] = node_dict["node_type"].name
        nodes.append(node_dict)

    edges = []
    for k, vs in scene_graph.get_edge_dict_copy().items():
        for v in vs:
            v.start
            v.rel
            v.end
            v.weight
            edges.append(
                {
                    "s_idx": k[0],
                    "e_idx": k[1],
                    "s_id": v.start.node_id,
                    "e_id": v.end.node_id,
                    "r": v.rel,
                    "w": v.weight,
                }
            )

    scene_graph_dict = {"edges": edges, "nodes": nodes}
    return scene_graph_dict


def load_graph_from_dict(graph_dict):
    g = SceneGraph()

    node_id_idx = defaultdict(dict)
    for i, node in enumerate(graph_dict["nodes"]):
        node = node.copy()
        node["node_type"] = getattr(NodeType, node["node_type"])
        n = ESDFNode(**node)
        node_id_idx[n.node_id][i] = n
        g.add_node(n)

    for e in graph_dict["edges"]:
        s_idx = e["s_idx"]
        e_idx = e["e_idx"]
        s_id = e["s_id"]
        e_id = e["e_id"]

        start_node = node_id_idx[s_id][s_idx]
        end_node = node_id_idx[e_id][e_idx]

        g.add_edge(SceneEdge(start_node, e["r"], end_node, e["w"]))

    return g
