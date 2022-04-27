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

from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from .utils import bresenham_raycast


class SceneESDFHandler:
    def __init__(
        self,
        esdf_path: str,
        esdf_thresh: Optional[float] = 0,
        esdf_downsample_radius: Optional[float] = 2,
        esdf_slice_range: Optional[Tuple[int, int]] = None,
        esdf_condense_operator: Optional[str] = "max",
    ) -> None:
        """Handles ESDF logic for RL / DSG relevant tasks.

        Args:
            esdf_path (str): Path to npy file of ESDF.
            esdf_pad_thresh (float): Value used to pad ESDF. This is needed when
                transforming the ESDF from ROS to Unity frames.
            esdf_downsample_radius (float): For efficiency, only consider
                points within `esdf_downsample_radius` of agent when
                freespace / line-of-sight checking. To ensure validity, this
                should be syncronized with the expected maximum distance.
        """
        self.__esdf_pts, self.__esdf_values = self._esdf_ros_to_unity_coords(
            esdf_path, esdf_thresh, esdf_slice_range, esdf_condense_operator
        )

        self.__scene_esdf_pts = None
        self.__scene_esdf_values = None
        self.__esdf_downsample_radius = esdf_downsample_radius
        self.__esdf_translation = np.zeros(2)
        self.__esdf_rotation = np.eye(2)

    def get_esdf_observation(
        self, rotation: float, mode="subsampled", format="min"
    ) -> np.ndarray:
        """Get ESDF Observation.

        Args:
            rotation (float): Desired rotation of ESDF
                w.r.t. world frame. In radians.
            mode (str): 'Subsampled' to crop ESDF around
                agent's current position. 'occupancy_grid'
                to give a stacked image of the ESDF and
                corresponding occupancy grid in the world frame.
            format (str): 'min': Min ESDF value across `esdf_slice_range`.
                'slices' to return all values in `esdf_slice_range`.

        Returns:
            np.ndarray: ESDF observation.
        """
        pos = self.unity_to_esdf_frame(np.zeros((1, 2)))[0]
        occupancy_grid = np.zeros(self.__esdf_img.shape)

        if mode == "subsampled":
            # downsample image before rotation to save time
            dch = 240
            dcw = 320

            # then crop final observation
            cw = 160
            ch = 120

            if format == "min":
                img = self.__esdf_img.copy()
            elif format == "slices":
                img = self.__esdf_slice_img.copy()
            else:
                raise ValueError(f"{format} not a valid option. Try [min, slices]")

            rot_deg = np.rad2deg(rotation)
            rot_deg += 90  # to image coords

            esdf_img, cy, cx = SceneESDFHandler.pad_img(img, pos[0], pos[1], dch, dcw)
            cropped_esdf = SceneESDFHandler.crop_img(esdf_img, cy, cx, dch, dcw)

            rot_img = scipy.ndimage.rotate(cropped_esdf, rot_deg)
            cropped_rot_esdf = SceneESDFHandler.crop_img(
                rot_img, rot_img.shape[0] // 2, rot_img.shape[1] // 2, ch, cw
            )

            obs = cropped_rot_esdf

        elif mode == "occupancy_grid":
            obs = np.stack((self.__esdf_img, occupancy_grid))

        # add channel axis if needed
        return obs[..., np.newaxis] if len(obs.shape) == 2 else obs

    @staticmethod
    def crop_img(img: np.ndarray, y: int, x: int, h: int, w: int) -> np.ndarray:
        return img[y - h // 2 : y + h // 2, x - w // 2 : x + w // 2]

    @staticmethod
    def pad_img(
        img: np.ndarray, cy: int, cx: int, ch: int, cw: int
    ) -> Tuple[np.ndarray, int, int]:
        """Pad `img` such that a crop around point (cy, cx)
        with height and width (ch, cw) can be taken.
        If padding is applied, update `cy` and `cx` to
        reflect the same relative position in `img`."""
        margins = np.array(
            [
                cy - ch // 2,
                img.shape[0] - (cy + ch // 2),
                cx - cw // 2,
                img.shape[1] - (cx + cw // 2),
            ]
        )

        if np.any(margins < 0):
            padding = np.maximum(-margins, 0)
            cy += padding[0]
            cx += padding[2]
            padding = (padding[:2], padding[2:])
            if len(img.shape) == 3:
                padding += (np.zeros(2, dtype=np.int32),)
            img = np.pad(img, padding, constant_values=0)  # defaut to 0 padding

        return img, cy, cx

    def _esdf_ros_to_unity_coords(
        self,
        esdf_path: str,
        esdf_thresh: Optional[float],
        esdf_slice_range: Optional[Tuple[int, int]] = None,
        condense_operator: Optional[str] = "max",
    ):
        with open(esdf_path, "rb") as f:
            esdf = np.load(f, encoding="latin1", allow_pickle=True)
            esdf = self._esdf_to_img(esdf)
            if esdf_slice_range:
                esdf = esdf[esdf_slice_range[0] : esdf_slice_range[1]]
                esdf[esdf < 0] = 0
            esdf_max = esdf.max(0) if condense_operator == "max" else esdf.min(0)

        self.__esdf_img = esdf_max
        self.__esdf_slice_img = esdf.transpose(1, 2, 0)
        return self._get_normalized_esdf_data(esdf_max, esdf_thresh)

    def get_esdf_img(self) -> np.ndarray:
        """Get ESDF as img."""
        return self.__esdf_img

    def bresenham_raycast(
        self, start_pt: np.ndarray, end_pts: np.ndarray
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        """Raycast from `start_pt` to each `end_pt` in `end_pts`
        via the Bresenham Line Algorithm.

        Args:
            start_pt (np.ndarray): (2, ) array of the starting point.
            end_pts (np.ndarray): (N, 2) array of `N` end points.

        Returns:
            Tuple[List[List[int]], List[Tuple[int, int]]]:
                - List of ESDF values on the line between
                `start_pt` and each `end_pt` in `end_pts`.
                - List of ESDF coordinates used for each raycast.
        """
        start_pt = self.unity_to_esdf_frame(start_pt[np.newaxis])[0]
        end_pts = self.unity_to_esdf_frame(end_pts)

        start_pt = start_pt[::-1]
        end_pts = end_pts[:, ::-1]
        self.start_pt = start_pt
        self.end_pts = end_pts
        esdf_vals = bresenham_raycast(start_pt, end_pts, self.__esdf_img)
        return esdf_vals

    def direct_esdf_lookup(self, pt: np.ndarray) -> float:
        """Get ESDF value of point `pt`.

        Args:
            pt (np.ndarray): (2, ) shape point in agent
                frame.
        """
        pt = self.unity_to_esdf_frame(pt[np.newaxis])[0]
        return self.__esdf_img[pt[0], pt[1]]

    def unity_to_esdf_frame(self, pts: np.ndarray) -> np.ndarray:
        """Convert `pts` from unity to ESDF frame.
        `pts` are expected to be in the agent frame.

        Args:
            pts (np.ndarray): shape (N, 2) array of `N` points.

        Returns:
            np.ndarray: shape (N, 2) array of transformed points.
        """
        # adjust for agent transform
        pts = np.matmul(self.__esdf_rotation.T, pts.T).T
        pts += self.__esdf_translation

        # adjust for scene transform
        pts -= self.__places_pts_min
        pts /= self.__places_pts_max - self.__places_pts_min

        # adjust for img scaling
        pts *= self.__esdf_max
        pts += self.__esdf_min

        # adjust for ros -> unity transform
        # pts = np.matmul(self.__ros_to_unity.T, pts.T).T
        return np.round(pts).astype(np.int32)

    def _get_normalized_esdf_data(
        self, esdf: np.ndarray, esdf_thresh: float
    ) -> np.ndarray:
        """Transform ESDF from ROS to Unity coordinates and normalize
        scale x/y values between [0, 1]."""
        ros_to_unity = np.array([[0, 1], [1, 0]])
        self.__ros_to_unity = ros_to_unity
        x, y = np.meshgrid(np.arange(esdf.shape[1]), np.arange(esdf.shape[0]))
        esdf_values = esdf[np.where(esdf > esdf_thresh)]
        esdf_pts = np.vstack(
            (x[np.where(esdf > esdf_thresh)], y[np.where(esdf > esdf_thresh)])
        ).T.astype(np.float64)

        esdf_pts = np.matmul(ros_to_unity, esdf_pts.T).T

        # scale esdf points between [0, 1]
        esdf_pts = esdf_pts.astype(np.float16)
        self.__esdf_min = esdf_pts.min(0)
        esdf_pts -= esdf_pts.min(0)
        self.__esdf_max = esdf_pts.max(0)
        esdf_pts /= esdf_pts.max(0)

        return esdf_pts, esdf_values

    def _esdf_to_img(self, esdf: np.ndarray) -> np.ndarray:
        """Get ESDF in 1 channel image format by taking the
        max slice value for each channel."""
        esdf = OrderedDict({float(k): v for k, v in dict(esdf.item()).items()})
        esdf = OrderedDict(sorted(esdf.items()))
        im = np.stack(list(esdf.values()))
        im[np.isnan(im)] = 0
        return im

    def set_esdf_unity_to_scene_graph_coords(
        self, places_pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform ESDF from Unity to scene graphframe.

        Note this may be different from the Unity frame, as
        the scene graph may be relative to the agent."""
        # get nonzero esdf points and transform to Unity coords
        # scale esdf to scene graph height / width
        self.__scene_esdf_pts = self.__esdf_pts * (
            places_pts.max(0) - places_pts.min(0)
        )

        # translate points to align w/ scene graph
        self.__scene_esdf_pts += places_pts.min(0)
        self.__scene_esdf_values = self.__esdf_values

        self.__places_pts_max = places_pts.max(0)
        self.__places_pts_min = places_pts.min(0)

    def transform_pts(self, translation: np.ndarray, rotation: np.ndarray) -> None:
        """Translate ESDF by `translation` and rotate by `rotation`."""
        self.__transformed_scene_esdf_pts = self.__scene_esdf_pts - translation
        self.__transformed_scene_esdf_pts = np.matmul(
            rotation, self.__transformed_scene_esdf_pts.T
        ).T
        self.__esdf_translation = translation
        self.__esdf_rotation = rotation

    def get_full_transformed_esdf(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get ESDF points and values in the unity scene frame."""
        return (self.__transformed_scene_esdf_pts, self.__scene_esdf_values)

    def downsample_transformed_pts(
        self, ref_pt: np.ndarray = np.array([0, 0]), thresh_m: Optional[float] = None
    ) -> None:
        """Keep ESDF points within `thresh_m` of `ref_pt`."""
        thresh_m = self.__esdf_downsample_radius if thresh_m is None else thresh_m
        in_range = np.where(
            np.linalg.norm(
                self.__transformed_scene_esdf_pts - ref_pt[np.newaxis], ord=2, axis=-1
            )
            < thresh_m
        )
        self.__downsampled_transformed_esdf_pts = self.__transformed_scene_esdf_pts[
            in_range
        ]
        self.__downsampled_transformed_esdf_values = self.__scene_esdf_values[in_range]

    def esdf_lookup(self, pt: np.ndarray) -> Tuple[float, float]:
        """Get distance to closest ESDF voxel, as well as the
        distance to that voxel."""
        if pt.shape == (2,):
            pt = pt[np.newaxis]

        closest_esdf_xy = np.linalg.norm(
            self.__downsampled_transformed_esdf_pts[np.newaxis] - pt[:, np.newaxis],
            ord=2,
            axis=-1,
        )

        # if there are no matches, assume `pt` is
        # outside of the ESDF and thus outside the scene
        if closest_esdf_xy.size == 0:
            return -1, -1

        return (
            np.min(closest_esdf_xy, -1),
            self.__downsampled_transformed_esdf_values[np.argmin(closest_esdf_xy, -1)],
        )

    def get_esdf_pts(self) -> np.ndarray:
        return self.__downsampled_transformed_esdf_pts

    def get_esdf_values(self) -> np.ndarray:
        return self.__downsampled_transformed_esdf_values

    def point_in_freespace(
        self,
        pt: np.ndarray,
        dist_t: Optional[float] = 0.05,
        value_t: Optional[float] = 0.1,
    ) -> bool:
        """True if `pt` is at least `value_t` from an obstacle."""
        closest_esdf_pt, closest_esdf_value = self.esdf_lookup(pt)
        if closest_esdf_pt == -1:
            return False
        return closest_esdf_pt < dist_t and closest_esdf_value > value_t

    def los_in_freespace(
        self,
        start_pt: np.ndarray,
        end_pt: np.ndarray,
        dist_t: Optional[float] = 1,
        value_t: Optional[float] = 0.1,
        eps=0.01,
        esdf="downsampled",
    ) -> bool:
        """True if line of sight between `start_pt` and `end_pt` does
        not intersect an obstacle."""
        num_steps = (np.linalg.norm(start_pt - end_pt, ord=2) / eps).astype(np.int32)
        pts = np.linspace(start_pt, end_pt, num=num_steps)

        # if batch, pts are of shape [raycast sample, batch, xy]
        # else, just [raycast sample, xy]
        if len(pts.shape) > 2 and pts.shape[1] > 1:
            pts = pts.transpose(1, 0, 2)  # -> [batch, raycast sample, xy]
            closest_esdf_pts, closest_esdf_values = self.esdf_lookup_batch(
                pts, esdf=esdf
            )
        else:
            closest_esdf_pts, closest_esdf_values = self.esdf_lookup(pts)

        return np.all(closest_esdf_pts < dist_t, axis=-1) * np.all(
            closest_esdf_values > value_t, axis=-1
        )

    def is_valid_node(
        self,
        node_pos: np.ndarray,
        ref_pt: Optional[np.ndarray] = np.array([0, 0]),
        step: Optional[float] = 0.1,
        dist_thresh: Optional[float] = 0.01,
        direct_esdf_lookup: Optional[bool] = False,
    ) -> bool:
        """True if 1) `node_pos` is in freespace and 2) line of
        sight between node and `ref_pt` does not intersect with
        an obstacle."""
        if direct_esdf_lookup:
            agent_node_in_fs = self.direct_esdf_lookup(node_pos)
            agent_node_in_fs = agent_node_in_fs > dist_thresh
        else:
            agent_node_in_fs = self.point_in_freespace(node_pos)

        # check for los  between agent and layer node
        if agent_node_in_fs and ref_pt is not None:
            if direct_esdf_lookup:
                out, _ = self.bresenham_raycast(ref_pt, node_pos[np.newaxis])
                agent_node_in_fs = out[0] > dist_thresh
            else:
                agent_node_in_fs = self.los_in_freespace(node_pos, ref_pt, eps=step)

        return agent_node_in_fs

    def get_valid_edges(
        self,
        ref_pt: np.ndarray,
        nodes: np.ndarray,
        layer_edges: List[int],
        step: Optional[float] = 0.1,
    ) -> np.ndarray:
        """Get all edges between nodes with a free
        line of sight to `ref_pt`.

        Notes:
            Deprecated in favor or `get_valid_edges_batch`. But
            keep until all experimentation is done.
        """
        freespace_edges = []
        for edge in layer_edges:
            los_free = self.los_in_freespace(ref_pt, nodes[edge, :2], eps=step)
            if los_free:
                freespace_edges.append(edge)
        freespace_edges = np.array(freespace_edges)
        return freespace_edges

    def get_valid_edges_batch(
        self,
        ref_pt: np.ndarray,
        nodes: np.ndarray,
        layer_edges: List[int],
        step: Optional[float] = 0.1,
        direct_esdf_lookup: Optional[bool] = False,
        value_t: Optional[float] = 0.01,
    ) -> np.ndarray:
        """Get all edges between nodes with a free
        line of sight to `ref_pt`."""
        if len(layer_edges) < 2:
            return []
        freespace_edges = []
        lookup_edges = nodes[layer_edges, :2]
        if direct_esdf_lookup:
            esdf_vals, _ = self.bresenham_raycast(ref_pt, lookup_edges)
            los_free = np.array(esdf_vals) > value_t
        else:
            los_free = self.los_in_freespace(ref_pt, lookup_edges, eps=step)
        layer_edges = np.array(layer_edges)
        freespace_edges = layer_edges[los_free]
        return freespace_edges

    def esdf_lookup_batch(
        self, pt: np.ndarray, esdf="downsampled"
    ) -> Tuple[float, float]:
        """Get distance to closest ESDF voxel, as well as the
        distance to that voxel."""
        # if there are no esdf values, assume `pt` is
        # outside of the ESDF and thus outside the scene
        if esdf == "downsampled":
            esdf_values = self.__downsampled_transformed_esdf_values
            esdf_pts = self.__downsampled_transformed_esdf_pts
        elif esdf == "full":
            esdf_values = self.__scene_esdf_values
            esdf_pts = self.__transformed_scene_esdf_pts
        if 0 in esdf_values.shape:
            return -1 * np.ones(pt.shape[0]), -1 * np.ones(pt.shape[0])

        esdf_pts = torch.tensor(esdf_pts[np.newaxis, np.newaxis]).cuda()
        pt = torch.tensor(pt[:, :, np.newaxis]).cuda()
        closest_esdf_xy = torch.norm(esdf_pts - pt, dim=-1)

        min_esdf_pt, min_esdf_arg = closest_esdf_xy.min(dim=-1)
        min_esdf_val = esdf_values[min_esdf_arg.cpu().numpy()]
        return min_esdf_pt.cpu().numpy(), min_esdf_val
