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

from tesse.msgs import Camera, SetCameraParametersRequest, SetCameraPositionRequest


def set_multiple_camera_params(
    tesse_gym,
    cameras=(Camera.RGB_LEFT, Camera.SEGMENTATION, Camera.DEPTH),
    height_in_pixels=240,
    width_in_pixels=320,
    field_of_view=60,
    near_clip_plane=0.05,
    far_clip_plane=50,
):
    """ Initialize gym environment camera settings.

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
    """ Initialize gym environment camera settings.

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
    """ Set gym environment camera parameters.

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


def _adjust_camera_position(tesse_gym, camera, x=0, y=0, z=0):
    """ Set gym environment camera position.

    Args:
        tesse_gym: (TesseGym): Gym environment.
        camera: (tesse.msgs.Camera): Camera whose parameters to adjust.
        x (int): X position.
        y (int): y position.
        z (int): z position.
    """
    tesse_gym.env.request(SetCameraPositionRequest(camera=camera, x=x, y=y, z=z))
