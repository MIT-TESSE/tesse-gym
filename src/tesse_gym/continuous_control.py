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

import defusedxml.ElementTree as ET
from scipy.spatial.transform import Rotation

from tesse.msgs import *
from tesse.utils import UdpListener

# gains 1: 150, 35, 1.6, 0.27
# gains 2: 200, 35, 1.6, 0.27


def get_attributes(root, element, *attributes):
    """ Get XML element attributes.

    Args:
        root (Element): XML root.
        element (str): Attribute element.
        *attributes (str): Attributes to fetch.

    Returns:
        List[str]: Requested attributes.
    """
    element = root.find(element)
    return [float(element.attrib[attrib]) for attrib in attributes]


def parse_metadata(metadata):
    """ Get position, orientation, velocity, and acceleration from metadata

    Args:
        metadata (str): TESSE metadata.

    Returns:
        Dict[str]: Dictionary containing position, orientation, velocity, and
            acceleration values.
    """
    data = {}
    root = ET.fromstring(metadata)

    x, y, z = get_attributes(root, "position", "x", "y", "z")
    data["position"] = {"x": x, "y": y, "z": z}

    x, y, z, w = get_attributes(root, "quaternion", "x", "y", "z", "w")
    data["quaternion"] = {"x": x, "y": y, "z": z, "w": w}
    data["rotation"] = Rotation((x, y, z, w)).as_euler("zxy")

    x_dot, y_dot, z_dot = get_attributes(root, "velocity", "x_dot", "y_dot", "z_dot")
    data["velocity"] = {"x_dot": x_dot, "y_dot": y_dot, "z_dot": z_dot}

    x_ang_dot, y_ang_dot, z_ang_dot = get_attributes(
        root, "angular_velocity", "x_ang_dot", "y_ang_dot", "z_ang_dot"
    )
    data["angular_velocity"] = {
        "x_ang_dot": x_ang_dot,
        "y_ang_dot": y_ang_dot,
        "z_ang_dot": z_ang_dot,
    }

    x_ddot, y_ddot, z_ddot = get_attributes(
        root, "acceleration", "x_ddot", "y_ddot", "z_ddot"
    )
    data["acceleration"] = {"x_ddot": x_ddot, "y_ddot": y_ddot, "z_ddot": z_ddot}

    x_ang_ddot, y_ang_ddot, z_ang_ddot = get_attributes(
        root, "angular_acceleration", "x_ang_ddot", "y_ang_ddot", "z_ang_ddot"
    )
    data["angular_acceleration"] = {
        "x_ang_ddot": x_ang_ddot,
        "y_ang_ddot": y_ang_ddot,
        "z_ang_ddot": z_ang_ddot,
    }

    data["time"] = float(root.find("time").text)
    data["collision"] = root.find("collision").attrib["status"].lower() == "true"

    return data


class ContinuousController:
    def __init__(
        self,
        env,
        threshold=np.array([0.05, 0.05, 0.01]),
        rate_threshold=np.array([0.01, 0.01, 0.01]),
        framerate=20,
        max_steps=100,
        pos_error_gain=150,
        pos_error_rate_gain=35,
        yaw_error_gain=1.6,
        yaw_error_rate_gain=0.27,
        udp_port=9004
    ):
        """ Initialize PD controller.

        Args:
            env (Env): Tesse Env object.
            threshold (np.ndarray): (x, z, rotation) error threshold to
                be considered at the goal point.
            rate_threshold (np.ndarray): (x velocity, z velocity, angular velocity)
                limit to be considered at goal.
            framerate (int): TESSE step mode framerate.
            max_steps (int): Maximum steps controller will take to reach goal.
            pos_error_gain (float): Position Proportional gain.
            pos_error_rate_gain (float): Position derivative gain.
            yaw_error_gain (float): Yaw proportional gain.
            yaw_error_rate_gain (float): Yaw derivative gain.
        """
        self.env = env
        self.threshold = threshold
        self.rate_threshold = rate_threshold
        self.env.send(SetFrameRate(framerate))  # Put into step mode
        self.max_steps = max_steps
        self.pos_error_gain = pos_error_gain
        self.pos_error_rate_gain = pos_error_rate_gain
        self.yaw_error_gain = yaw_error_gain
        self.yaw_error_rate_gain = yaw_error_rate_gain

        self.goal = []

        self.last_metadata = None
        self.udp_listener = UdpListener(port=udp_port, rate=200)
        self.udp_listener.subscribe("catch_metadata", self.catch_udp_broadcast)

        self.udp_listener.start()

    def catch_udp_broadcast(self, udp_metadata):
        self.last_metadata = udp_metadata

    def transform(self, translate_x=0.0, translate_z=0.0, rotate_y=0.0):
        """ Apply desired transform via force commands.

        Args:
            translate_x (float): Desired x position relative to agent.
            translate_z (float): Desired z position relative to agent.
            rotate_y (float): Desired rotation (in radians) relative to agent.
         """
        data = self.get_data()
        self.set_goal(data, translate_x, translate_z, rotate_y)

        # Apply controls until at goal point, a collision occurs, or max steps reached
        i = 0
        while not self.at_goal(data) and i < self.max_steps:
            self.control(data)
            data = self.get_data()
            i += 1
            # check for collisions after at least two steps
            if data["collision"] and i > 1:
                break

        self.set_goal(data)

    def get_data(self):
        """ Gets current data for agent """
        if self.last_metadata is None:
            response = self.env.request(MetadataRequest()).metadata
        else:
            response = self.last_metadata
        return parse_metadata(response)

    def set_goal(self, data, translate_x=0.0, translate_z=0.0, rotate_y=0.0):
        """ Sets the goal for the controller via creating a waypoint based
        on the desired transform.

        Args:
            data (Dict[str]): Agent's position, orientation, velocity,
                and acceleration.
            translate_x (float): Desired x position relative to agent.
            translate_z (float): Desired z position relative to agent.
            rotate_y (float): Desired rotation (in radians) relative to agent.
        """
        # Update goal point
        yaw = data["rotation"][2]
        x = (
            data["position"]["x"]
            + translate_x * np.cos(-yaw)
            - translate_z * np.sin(-yaw)
        )
        z = (
            data["position"]["z"]
            + translate_x * np.sin(-yaw)
            + translate_z * np.cos(-yaw)
        )
        self.goal = np.array([x, z, yaw + rotate_y])

    def at_goal(self, data):
        """ Returns True if at the goal location within the threshold.

        Args:
            data (Dict[str, Dict[str, str]]): Agent's position, orientation, velocity,
                and acceleration.
        """
        # check position
        current = np.array(
            [data["position"]["x"], data["position"]["z"], data["rotation"][2]]
        )
        error = current - self.goal
        error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi  # wrap to pi

        current_rate = np.array(
            [data['velocity']['x_dot'], data['velocity']['z_dot'], data['angular_velocity']['y_ang_dot']]
        )

        return np.all(np.abs(error) < self.threshold) and np.all(np.abs(current_rate) < self.rate_threshold)

    def control(self, data):
        """ Applies PD-control to move to the goal point.

        Args:
            data (Dict[str, Dict[str,str]]): Agent's position, orientation, velocity,
                and acceleration.
        """
        # First, calculate position errors and a force in x- and z- to apply
        x_error = self.goal[0] - data["position"]["x"]
        z_error = self.goal[1] - data["position"]["z"]

        # Rotate errors into body coordinates
        yaw = data["rotation"][2]
        z_error_body = z_error * np.cos(-yaw) - x_error * np.sin(-yaw)
        x_error_body = z_error * np.sin(-yaw) + x_error * np.cos(-yaw)

        z_error_body_rate = -1 * data["velocity"]["z_dot"]
        x_error_body_rate = -1 * data["velocity"]["x_dot"]

        force_z = (
            self.pos_error_gain * z_error_body
            + self.pos_error_rate_gain * z_error_body_rate
        )
        force_x = (
            self.pos_error_gain * x_error_body
            + self.pos_error_rate_gain * x_error_body_rate
        )

        # Second, calculate yaw error assuming we want to point to where we are going
        yaw_error = self.goal[2] - yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # wrap to pi

        yaw_error_rate = -1 * data["angular_velocity"]["y_ang_dot"]
        torque_y = (
            self.yaw_error_gain * yaw_error + self.yaw_error_rate_gain * yaw_error_rate
        )

        self.env.send(StepWithForce(force_z, torque_y, force_x))
