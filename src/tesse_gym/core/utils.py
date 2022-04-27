###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2019 Massachusetts Institute of Technology.
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

from collections import namedtuple
from typing import Union

from tesse.msgs import (
    Camera,
    DataResponse,
    SetCameraParametersRequest,
    SetCameraPositionRequest,
)

NetworkConfig = namedtuple(
    "NetworkConfig",
    [
        "simulation_ip",
        "own_ip",
        "position_port",
        "metadata_port",
        "image_port",
        "step_port",
    ],
    defaults=("localhost", "localhost", 9000, 9001, 9002, 9005),
)


def get_network_config(
    simulation_ip="localhost",
    own_ip="localhost",
    base_port=9000,
    worker_id=0,
    n_ports=6,
):
    """Get a TESSE network configuration instance.

    Args:
        simulation_ip (str): TESSE IP address.
        own_ip (str): Local IP address.
        base_port (int): Starting connection port. It is assumed the rest of the ports
            follow sequentially.
        worker_id (int): Worker ID of this Gym instance. Ports are staggered by ID.
        n_ports (int): Number of ports allocated to each TESSE instance.

    Returns:
        NetworkConfig: NetworkConfig object.
    """
    return NetworkConfig(
        simulation_ip=simulation_ip,
        own_ip=own_ip,
        position_port=base_port + worker_id * n_ports,
        metadata_port=base_port + worker_id * n_ports + 1,
        image_port=base_port + worker_id * n_ports + 2,
        step_port=base_port + worker_id * n_ports + 5,
    )


def set_all_camera_params(
    tesse_gym,
    cameras=(
        Camera.RGB_LEFT,
        Camera.SEGMENTATION,
        Camera.DEPTH,
    ),  # , Camera.THIRD_PERSON),
    height_in_pixels=240,
    width_in_pixels=320,
    field_of_view=60,
    near_clip_plane=0.05,
    far_clip_plane=50,
):
    """Initialize gym environment camera settings.

    Args:
        tesse_gym (TesseGym): Gym environment.
        cameras (List[tesse.msgs.Camera]): Cameras whose parameters to change.
        height_in_pixels (int): Camera height in pixels.
        width_in_pixels (int): Camera width in pixels.
        field_of_view (int): Camera field of view in degrees.
        near_clip_plane (Union[float, int]): Camera near clipping plane.
        far_clip_plane (Union[float, int]): Camera far clipping plane.
    """
    for camera in cameras:
        set_camera_params(
            tesse_gym,
            camera,
            height_in_pixels,
            width_in_pixels,
            field_of_view,
            near_clip_plane,
            far_clip_plane,
        )


def set_camera_params(
    tesse_gym,
    camera,
    height_in_pixels,
    width_in_pixels,
    field_of_view,
    near_clip_plane,
    far_clip_plane,
):
    """Initialize gym environment camera settings.

    Args:
        tesse_gym (TesseGym): Gym environment.
        camera (tesse.msgs.Camera): Camera whose parameters to change.
        height_in_pixels (int): Camera height in pixels.
        width_in_pixels (int): Camera width in pixels.
        field_of_view (int): Camera field of view in degrees.
        near_clip_plane (Union[float, int]): Camera near clipping plane.
        far_clip_plane (Union[float, int]): Camera far clipping plane.
    """
    _adjust_camera_params(
        tesse_gym,
        camera,
        height_in_pixels,
        width_in_pixels,
        field_of_view,
        near_clip_plane,
        far_clip_plane,
    )
    _adjust_camera_position(tesse_gym, camera)


def _adjust_camera_params(
    tesse_gym,
    camera,
    height_in_pixels,
    width_in_pixels,
    field_of_view,
    near_clip_plane,
    far_clip_plane,
):
    """Set gym environment camera parameters.

    Args:
        tesse_gym (TesseGym): Gym environment.
        cameras (tesse.msgs.Camera): Camera whose parameters to change.
        height_in_pixels (int): Camera height in pixels.
        width_in_pixels (int): Camera width in pixels.
        field_of_view (int): Camera field of view in degrees.
        near_clip_plane (Union[float, int]): Camera near clipping plane.
        far_clip_plane (Union[float, int]): Camera far clipping plane.
    """
    tesse_gym.env.request(
        SetCameraParametersRequest(
            camera=camera,
            height_in_pixels=height_in_pixels,
            width_in_pixels=width_in_pixels,
            field_of_view=field_of_view,
            near_clip_plane=near_clip_plane,
            far_clip_plane=far_clip_plane,
        )
    )


def _adjust_camera_position(tesse_gym, camera, x=-0.05, y=0, z=0):
    """Set gym environment camera position.

    Args:
        tesse_gym: (TesseGym): Gym environment.
        camera: (tesse.msgs.Camera): Camera whose parameters to adjust.
        x (int): X position.
        y (int): y position.
        z (int): z position.
    """
    tesse_gym.env.request(SetCameraPositionRequest(camera=camera, x=x, y=y, z=z))


def response_nonetype_check(obs: Union[DataResponse, None]) -> DataResponse:
    """Check that data from the sim is not `NoneType`.

    `obs` being `NoneType` indicates that data could
    not be read from TESSE. Raise an exception if this
    is the case.

    Args:
        obs (Union[DataResponse, None]): Response from the simulator.

    Returns:
        DataResponse: `obs` if `obs` is not `None`.

    Raises:
        TesseConnectionError
    """
    if obs is None:
        raise TesseConnectionError()
    elif hasattr(obs, "metadata") and obs.metadata is None:
        raise TesseConnectionError()
    return obs


class TesseConnectionError(Exception):
    def __init__(self):
        """ Indicates data cannot be read from TESSE. """
        self.message = (
            "Cannot receive data from the simulator. "
            "The connection is blocked or the simulator is not running. "
        )

    def __str__(self):
        return self.message
