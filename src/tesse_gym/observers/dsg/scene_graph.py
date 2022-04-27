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

from typing import Any, Dict, Tuple, Union

import numpy as np
from gym import spaces
from tesse_gym.actions.discrete import DiscreteNavigationMapper
from tesse_gym.core.observations import ObservationConfig, setup_observations
from tesse_gym.core.tesse_gym import TesseGym
from tesse_gym.observers.dsg.action_layer import ActionLayer
from tesse_gym.observers.dsg.esdf import SceneESDFHandler
from tesse_gym.observers.image_observer import TesseImageObserver
from tesse_gym.observers.observer import Observer

from .accumulation_model import DSGAccumulationModel


class DSGObserver(Observer):
    def __init__(
        self,
        scene_graph: Dict[str, str],
        use_esdf_data: bool,
        agent_frame_params: str,
        esdf_data: Dict[str, str],
        mark_visited_nodes: bool = True,
        coordinate_system: str = "cartesian",
        agent_frame_type: str = "node_frame",
        graph_frame: str = "current_pose",
        esdf_data_format: str = "min",
        accumulate_graph: bool = True,
        agent_frame_edge_filter_ref: str = "layer_node",
        agent_frame_direct_esdf_edge_check: bool = False,
        esdf_value_thresh: float = 0,
        esdf_downsample_dist: float = 2,
        esdf_slice_range: Tuple[int, int] = [11, 21],
        esdf_condense_operator: str = "min",
    ):
        """Get TesseGym handler for scene graph.

        TODO(ZR) clean up action layer / remove some
            args before documentation.
        """
        self.scene_graph_handlers = {}
        self.__mark_visited_nodes = mark_visited_nodes
        self.__n_visited_nodes = 0
        self.__graph_frame = graph_frame

        assert self.__graph_frame in ("spawn_pose", "current_pose")

        # esdf data
        self.provide_esdf_data = use_esdf_data
        if self.provide_esdf_data:
            # get esdf slices, or min across slices
            self.esdf_data_format = esdf_data_format

        # coordinate system
        self.coord_system = coordinate_system

        # action layer
        self.__agent_frame_handler = ActionLayer(
            agent_frame_type,
            agent_frame_params,
            agent_frame_edge_filter_ref,
            direct_esdf_lookup_node=agent_frame_direct_esdf_edge_check,
            direct_esdf_lookup_edge=agent_frame_direct_esdf_edge_check,
        )

        esdf_handler = self._init_esdf_scene_handler(
            esdf_paths=esdf_data,
            esdf_value_thresh=esdf_value_thresh,
            esdf_downsample_dist=esdf_downsample_dist,
            esdf_slice_range=esdf_slice_range,
            esdf_condense_operator=esdf_condense_operator,
        )

        # scene graphs
        for i, fp in scene_graph.items():
            self.scene_graph_handlers[int(i)] = DSGAccumulationModel(
                fp,
                esdf_handler=esdf_handler[int(i)] if esdf_handler is not None else None,
                accumulate_graph=accumulate_graph,
            )
        self.__esdf_handler = esdf_handler

        self._setup_edge_padding()
        self.observation_space = self._setup_observations()
        self.current_scene = None

    def _setup_observations(self) -> spaces.Dict:
        """Add DSG to observation space."""
        custom_obs = {
            "GRAPH_NODES": (-np.Inf, np.Inf, self.max_node_shape, np.float64),
            "GRAPH_EDGES": (
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).max,
                self.max_edge_shape,
                np.int64,
            ),
            "GRAPH_NODE_SHAPE": (-np.Inf, np.Inf, (2,), np.float32),
            "GRAPH_EDGE_SHAPE": (-np.Inf, np.Inf, (2,), np.float32),
        }
        if self.provide_esdf_data is not None:
            # TOOD (ZR) remove hardcoding
            if self.esdf_data_format == "min":
                esdf_obs_shape = (120, 160, 1)
            elif self.esdf_data_format == "min_semantic":
                esdf_obs_shape = (120, 160, 2)
            elif self.esdf_data_format == "slices":
                esdf_obs_shape = (120, 160, 10)  # TODO(zr) remove hardcoding
            else:
                raise ValueError(f"{self.esdf_data_format} not supported")
            custom_obs["ESDF"] = (-np.Inf, np.Inf, esdf_obs_shape, np.float64)

        obs_names = []
        obs_spaces = []
        for k, (custom_min, custom_max, custom_shape, dtype) in custom_obs.items():
            obs_names.append(k)
            obs_spaces.append(spaces.Box(custom_min, custom_max, custom_shape, dtype))

        observation_space = spaces.Dict(dict(zip(obs_names, obs_spaces)))
        return observation_space

    def _setup_edge_padding(self) -> None:
        """Determine node / edge tensor padding needed
        for batched rllib observations."""
        self.max_node_shape = np.array(
            [
                sgh.get_pyg_scene_graph().x.numpy().shape
                for sgh in self.scene_graph_handlers.values()
            ]
        ).max(0)
        self.max_edge_shape = np.array(
            [
                sgh.get_pyg_scene_graph().edge_index.numpy().shape
                for sgh in self.scene_graph_handlers.values()
            ]
        ).max(0)

        # pad edges
        if self.__agent_frame_handler is not None:
            self.max_node_shape[0] += self.__agent_frame_handler.new_nodes
            self.max_edge_shape[1] = (1.5 * self.max_edge_shape[1]).astype(np.int32)

    def _init_args(self, kwargs: Dict[str, Any]) -> None:
        """Get scene graph specific keyword arguments.

        Args:
            kwargs (Dist[str, Any]): keywork arguments with
                the following keys
                - scene_graph (Union[List[str], str])
                - mark_visited_nodes (bool)
                - graph_frame (str)
                - coordinate_system (Optional[str])
                - agent_frame_type (Optional[None, str])
        """
        assert "scene_graph" in kwargs.keys()

    def _init_esdf_scene_handler(
        self,
        esdf_paths: Dict[str, str],
        esdf_value_thresh: float,
        esdf_downsample_dist: float,
        esdf_slice_range: Tuple[int, int],
        esdf_condense_operator: str,
    ) -> Union[None, Dict[str, SceneESDFHandler]]:
        """Get ESDF Scene handlers to perform operations
        over the ESDF needed for DSG observation constructions
        (e.g., free-space checking).

        Args:
            esdf_paths (Dict[str, str]): Dictionary mapping
                scene IDs to corresponding ESDF paths.
            esdf_value_thresh (float): Consider all ESDF values
                above this free-space.
            esdf_downsample_dist (float): Distance used to
                downsample ESDF around agent.
            esdf_slide_range (Tuple[int, int]): Range of ESDF
                slices to consider for free-space checking.
            esdf_concense_operator (str): TODO(ZR) should
                remove this option
        """
        esdf_handlers = {}

        for scene, esdf_path in esdf_paths.items():
            esdf_handlers[int(scene)] = SceneESDFHandler(
                esdf_path,
                esdf_thresh=esdf_value_thresh,
                esdf_downsample_radius=esdf_downsample_dist,
                esdf_slice_range=esdf_slice_range,
                esdf_condense_operator=esdf_condense_operator,
            )
        return esdf_handlers

    def get_visited_nodes(self) -> int:
        return self.__n_visited_nodes

    def get_observation_config(self) -> ObservationConfig:
        return self.__observation_config

    def reset_scene_graph_handler(self, scene_id: int) -> None:
        """Reset episode history for `scene_id`"""
        self.scene_graph_handlers[scene_id].reset()

    def get_dsg_observation(
        self,
        pose_rotation: np.ndarray,
        pose_translation: np.ndarray,
        current_scene: int,
        visited_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Form DSG observation.

        Args:
            pose_rotation (np.ndarray): Shape (2, 2) array of
                2D rotation applied to DSG.
            pose_translation (np.ndarray): Shape (2, ) array of
                translation applied to DSG.
            current_scene (int): Scene ID.
            visited_points (np.ndarray): Shape (N, 2) array
                of visited points on the 2D plane.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - DSG nodes
            - DSG edges
            - DSG node shapes (pre-padding)
            - DSG edge shapes (pre-padding)
        """
        pose_rotation = pose_rotation
        pose_translation = pose_translation

        nodes, edges, node_shape, edge_shape = self.scene_graph_handlers[
            current_scene
        ].get_scene_graph(
            node_pad_size=self.max_node_shape,
            edge_pad_size=self.max_edge_shape,
            pose_translation=pose_translation,
            pose_rotation=pose_rotation,
            visited_points=visited_points,
            polar_coords=self.coord_system == "polar",
            agent_frame_handler=self.__agent_frame_handler,
        )
        if self.__mark_visited_nodes:
            self.__n_visited_nodes = nodes[: node_shape[0], 8].sum()

        return nodes, edges, node_shape, edge_shape

    def get_esdf_observation(self, rotation: np.ndarray, scene_id: int) -> np.ndarray:
        """Get ESDF observation.

        Args:
            rotation (np.ndarray): Shape (2, 2) 2D rotation matrix.
            scene_id (int): Scene from which to get the ESDF.
        """
        return self.__esdf_handler[scene_id].get_esdf_observation(
            rotation=rotation,
            format=self.esdf_data_format,
        )

    def observe(self, env_info: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        initial_pose = env_info["initial_pose"]
        relative_pose = env_info["relative_pose"]
        self.current_scene = env_info["scene"]
        episode_trajectory = env_info["episode_trajectory"]

        world_to_spawn_rot = TesseGym.get_2d_rotation_mtrx(-1 * initial_pose[2])

        dsg_rotation = TesseGym.get_2d_rotation_mtrx(relative_pose[2] + initial_pose[2])

        dsg_translation = initial_pose[:2] + np.matmul(
            world_to_spawn_rot.T, relative_pose[:2]
        )

        (nodes, edges, node_shape, edge_shape,) = self.get_dsg_observation(
            pose_rotation=dsg_rotation,
            pose_translation=dsg_translation,
            current_scene=self.current_scene,
            visited_points=episode_trajectory[:, :2],
        )

        observation = {
            "GRAPH_NODES": nodes,
            "GRAPH_EDGES": edges,
            "GRAPH_NODE_SHAPE": node_shape.astype(np.float32),
            "GRAPH_EDGE_SHAPE": edge_shape.astype(np.float32),
        }

        if self.provide_esdf_data:
            observation["ESDF"] = self.get_esdf_observation(
                rotation=(initial_pose[2] + relative_pose[2]),
                scene_id=self.current_scene,
            )

        return observation

    def reset(self):
        if self.current_scene:
            self.reset_scene_graph_handler(self.current_scene)


def populate_scene_graph_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Populate scene graph observer config and provide some defaults"""
    return {
        "scene_graph": config.pop("scene_graph"),
        "mark_visited_nodes": config.pop("mark_visited_nodes", True),
        "graph_frame": config.pop("graph_frame", "current_pose"),
        "use_esdf_data": config.pop("use_esdf_data"),
        "esdf_data_format": config.pop("esdf_data_format", "min"),
        "coordinate_system": config.pop("coordinate_system", "cartesian"),
        "agent_frame_type": config.pop("agent_frame_type", "node_frame"),
        "agent_frame_params": config.pop("agent_frame_params", [8, [1], [1]]),
        "agent_frame_edge_filter_ref": config.pop(
            "agent_frame_edge_filter_ref", "layer_node"
        ),
        "agent_frame_direct_esdf_edge_check": config.pop(
            "agent_frame_direct_esdf_edge_check", False
        ),
        "accumulate_graph": config.pop("accumulate_graph", True),
        "esdf_data": config.pop("esdf_data"),
        "esdf_value_thresh": config.pop("esdf_value_thresh", 0),
        "esdf_downsample_dist": config.pop("esdf_downsample_dist", 2),
        "esdf_slice_range": config.pop("esdf_slice_range", [11, 21]),
        "esdf_condense_operator": config.pop("esdf_condense_operator", "min"),
    }


def get_scene_graph_task(BaseTask: TesseGym, **kwargs) -> TesseGym:
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
    obs_modalities, obs_space = setup_observations(kwargs["observation_config"])
    image_observer = TesseImageObserver(obs_modalities, obs_space)
    dsg_observer = DSGObserver(**populate_scene_graph_config(kwargs))
    kwargs["observers"] = {"image": image_observer, "dsg": dsg_observer}
    kwargs["action_mapper"] = DiscreteNavigationMapper()
    return BaseTask(**kwargs)
