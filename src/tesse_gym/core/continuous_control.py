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

from collections import namedtuple
from typing import Callable, List, Optional, Tuple
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
from scipy.spatial.transform import Rotation

from tesse.msgs import *
from tesse.utils import UdpListener


# gains 1: 150, 35, 1.6, 0.27
# gains 2: 200, 35, 1.6, 0.27


# Hold agent's state
Position = namedtuple("Position", ["x", "y", "z"])
Quaternion = namedtuple("Quaternion", ["x", "y", "z", "w"])
Rot = namedtuple("Rotation", ["roll", "pitch", "yaw"])
Velocity = namedtuple("Velocity", ["x", "y", "z"])
AngularVelocity = namedtuple("AngularVelocity", ["x", "y", "z"])
Acceleration = namedtuple("Acceleration", ["x", "y", "z"])
AngularAcceleration = namedtuple("AngularAcceleration", ["x", "y", "z"])
AgentState = namedtuple(
    "AgentState",
    [
        "position",
        "quaternion",
        "rotation",
        "velocity",
        "angular_velocity",
        "acceleration",
        "angular_acceleration",
        "time",
        "collision",
    ],
)

# The agent is considered in a collision if
# (1) It has applied force in the forward direction above `force_limit`
# (2) The current position error is above `position_error_limit`
# (3) The net error change is below `position_error_change_limit`
#
# Essentially checking if force has been applied to correct an error,
# but the agent has not moved
CollisionThresholds = namedtuple(
    "CollisionThresholds",
    ["force_limit", "position_error_limit", "position_error_change_limit"],
    defaults=(1e-2, 1e-2, 1e-5),
)


# Gains for a standard Proportional-Derivative controller
PDGains = namedtuple(
    "PDGains",
    ["pos_error_gain", "pos_error_rate_gain", "yaw_error_gain", "yaw_error_rate_gain"],
    defaults=(150, 35, 1.6, 0.27),
)


def get_attributes(
    root: Element, element: str, attributes: Tuple[str, ...]
) -> List[float]:
    """ Get XML element attributes.

    Args:
        root (Element): XML root.
        element (str): Attribute element.
        attributes (Tuple[str, ...]): Attributes to fetch.

    Returns:
        List[str]: Requested attributes.
    """
    element = root.find(element)
    return [float(element.attrib[attrib]) for attrib in attributes]


def parse_metadata(metadata: str) -> AgentState:
    """ Get position, orientation, velocity, and acceleration from metadata

    Args:
        metadata (str): TESSE metadata.

    Returns:
        AgentState: Object containing position, orientation, velocity, and
            acceleration values.
    """
    root = ET.fromstring(metadata)

    position = Position(*get_attributes(root, "position", ("x", "y", "z")))

    x, y, z, w = get_attributes(root, "quaternion", ("x", "y", "z", "w"))
    quat = Quaternion(x, y, z, w)
    rot = Rot(*Rotation((x, y, z, w)).as_euler("zxy"))

    velocity = Velocity(*get_attributes(root, "velocity", ("x_dot", "y_dot", "z_dot")))

    angular_velocity = AngularVelocity(
        *get_attributes(
            root, "angular_velocity", ("x_ang_dot", "y_ang_dot", "z_ang_dot")
        )
    )

    acceleration = Acceleration(
        *get_attributes(root, "acceleration", ("x_ddot", "y_ddot", "z_ddot"))
    )

    angular_acceleration = AngularAcceleration(
        *get_attributes(
            root, "angular_acceleration", ("x_ang_ddot", "y_ang_ddot", "z_ang_ddot")
        )
    )

    state = AgentState(
        position=position,
        quaternion=quat,
        rotation=rot,
        velocity=velocity,
        angular_velocity=angular_velocity,
        acceleration=acceleration,
        angular_acceleration=angular_acceleration,
        time=float(root.find("time").text),
        collision=root.find("collision").attrib["status"].lower() == "true",
    )

    return state


class ContinuousController:
    udp_listener_rate = 200  # frequency in hz at which to listen to UDP broadcasts
    collision_limit = 5  # break current control loop after this many collisions

    def __init__(
        self,
        *,
        env,
        position_error_threshold: Optional[np.ndarray] = np.array([0.05, 0.05, 0.01]),
        velocity_error_threshold: Optional[np.ndarray] = np.array([0.01, 0.01, 0.01]),
        framerate: Optional[int] = 20,
        max_steps: Optional[int] = 100,
        pd_gains: Optional[PDGains] = PDGains(),
        udp_port: Optional[int] = 9004,
        collision_thresholds: Optional[CollisionThresholds] = CollisionThresholds(),
    ):
        """ Initialize PD controller.

        Args:
            env (Env): Tesse Env object.
            position_error_threshold (Optional[np.ndarray]): (x, z, rotation) error threshold to
                be considered at the goal point.
            velocity_error_threshold (Optional[np.ndarray]): (x velocity, z velocity, angular velocity)
                limit to be considered at goal.
            framerate (Optional[int]): TESSE step mode framerate.
            max_steps (Optional[int]): Maximum steps controller will take to reach goal.
            pd_gains (Optional[PDGains]): Proportional-Derivative controller gains.
            udp_port (Optional[int]): Port on which to listen for metadata UDP broadcasts.
            collision_thresholds (Optional[CollisionThresholds]): Thresholds defining when an agent is in a
                collision. See `_in_collision` for details.
        """
        self.env = env
        self.position_error_threshold = position_error_threshold
        self.velocity_error_threshold = velocity_error_threshold
        self.env.send(SetFrameRate(framerate))  # Put sim into step mode
        self.max_steps = max_steps
        self.pd_gains = pd_gains
        self.collision_thresholds = collision_thresholds

        self.goal = []

        self.last_metadata = None
        self.udp_listener = UdpListener(port=udp_port, rate=self.udp_listener_rate)
        self.udp_listener.subscribe("catch_metadata", self.catch_udp_broadcast)

        self.udp_listener.start()

    def catch_udp_broadcast(self, udp_metadata: Callable[[str], None]) -> None:
        """ Catch UDP metadata broadcast from TESSE. """
        self.last_metadata = udp_metadata

    def transform(
        self, translate_x: float = 0.0, translate_z: float = 0.0, rotate_y: float = 0.0
    ) -> None:
        """ Apply desired transform via force commands.

        Args:
            translate_x (float): Desired x position relative to agent.
            translate_z (float): Desired z position relative to agent.
            rotate_y (float): Desired rotation (in radians) relative to agent.
        """
        data = self.get_data()
        self.set_goal(data, translate_x, translate_z, rotate_y)

        last_z_err, last_z_rate_err = 0, 0
        collision_count = 0
        n_steps = 0

        # Apply controls until at goal point, a collision occurs, or max steps reached
        while not self.at_goal(data) and n_steps < self.max_steps:
            force_z, z_error = self.control(data)
            data = self.get_data()

            if self._in_collision(force_z, z_error, last_z_err):
                collision_count += 1

            if collision_count > self.collision_limit:
                break

            last_z_err = z_error
            n_steps += 1

        self.set_goal(data)

    def _in_collision(
        self, force_z: float, z_pos_error: float, last_z_pos_err: float
    ) -> bool:
        """ Check if agent is in collision with an object at a given step.

        Count collision if
            (1) There is error in the forward direction
            (2) Force has been applied in the forward direction
            (3) The agent has not moved in the forward direction

        Args:
            force_z (float): Force applied in the z direction at
                current step.
            z_pos_error (float): Z position error at current step.
            last_z_pos_err (float): Z position error at previous
                step.

        Returns:
            bool: True if the agent is in a collision.
        """
        return (
            np.sign(force_z) == np.sign(z_pos_error)
            and np.abs(z_pos_error) > self.collision_thresholds.position_error_limit
            and np.abs(force_z) > self.collision_thresholds.force_limit
            and np.abs(z_pos_error - last_z_pos_err)
            < self.collision_thresholds.position_error_change_limit
        )

    def get_data(self) -> AgentState:
        """ Gets agent's most recent data. """
        if self.last_metadata is None:
            response = self.env.request(MetadataRequest()).metadata
        else:
            response = self.get_broadcast_metadata()
        return parse_metadata(response)

    def set_goal(
        self,
        data: AgentState,
        translate_x: float = 0.0,
        translate_z: float = 0.0,
        rotate_y: float = 0.0,
    ) -> None:
        """ Sets the goal for the controller via creating a waypoint based
        on the desired transform.

        Args:
            data (AgentState): Agent's position, orientation, velocity,
                and acceleration.
            translate_x (float): Desired x position relative to agent.
            translate_z (float): Desired z position relative to agent.
            rotate_y (float): Desired rotation (in radians) relative to agent.
        """
        # Update goal point
        yaw = data.rotation.yaw
        x = data.position.x + translate_x * np.cos(-yaw) - translate_z * np.sin(-yaw)
        z = data.position.z + translate_x * np.sin(-yaw) + translate_z * np.cos(-yaw)
        self.goal = np.array([x, z, yaw + rotate_y])

    def at_goal(self, data: AgentState) -> bool:
        """ Returns true if within position and velocity thresholds.

        Args:
            data (AgentState): Object with agent's position, orientation, velocity,
                and acceleration.
        """
        # check position
        current = np.array([data.position.x, data.position.z, data.rotation.yaw])
        error = current - self.goal
        error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi  # wrap to pi

        current_rate = np.array(
            [data.velocity.x, data.velocity.z, data.angular_velocity.y,]
        )

        return np.all(np.abs(error) < self.position_error_threshold) and np.all(
            np.abs(current_rate) < self.velocity_error_threshold
        )

    @staticmethod
    def _wrap_angle(ang: float) -> float:
        """ Wrap angle between [-2*pi, 2*pi]

        Args:
            ang (float): Angle in radians.

        Returns:
            float: Angle wrapped between [-2*pi, 2*pi].
        """
        return (ang + np.pi) % (2 * np.pi) - np.pi

    def control(self, data: AgentState) -> Tuple[float, float]:
        """ Applies PD-control to move to the goal point.

        Args:
            data (AgentState): Agent's position, orientation, velocity,
                and acceleration.

        Returns:
            Tuple[float, float]: Applied force and position error
                in the z direction.
        """
        # First, calculate position errors and a force in x- and z- to apply
        x_error = self.goal[0] - data.position.x
        z_error = self.goal[1] - data.position.z

        # Rotate errors into body coordinates
        yaw = data.rotation.yaw
        z_error_body = z_error * np.cos(-yaw) - x_error * np.sin(-yaw)
        x_error_body = z_error * np.sin(-yaw) + x_error * np.cos(-yaw)

        z_error_body_rate = -1 * data.velocity.z
        x_error_body_rate = -1 * data.velocity.x

        force_z = (
            self.pd_gains.pos_error_gain * z_error_body
            + self.pd_gains.pos_error_rate_gain * z_error_body_rate
        )
        force_x = (
            self.pd_gains.pos_error_gain * x_error_body
            + self.pd_gains.pos_error_rate_gain * x_error_body_rate
        )

        # Second, calculate yaw error assuming we want to point to where we are going
        yaw_error = self.goal[2] - yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # wrap to pi

        yaw_error_rate = -1 * data.angular_velocity.y
        torque_y = (
            self.pd_gains.yaw_error_gain * yaw_error
            + self.pd_gains.yaw_error_rate_gain * yaw_error_rate
        )

        self.env.send(StepWithForce(force_z, torque_y, force_x))
        return force_z, z_error_body

    def get_current_time(self) -> float:
        """ Get current sim time. """
        # TODO(ZR) specific logic for this needs to be figured out ``
        if self.last_metadata is None:
            raise ValueError("Cannot get TESSE time, metadata is `NoneType`")
        else:
            return float(ET.fromstring(self.last_metadata).find("time").text)

    def get_broadcast_metadata(self) -> str:
        """ Get metadata provided by TESSE UDP broadcasts. """
        return self.last_metadata

    def close(self):
        """ Called upon destruction, join UDP listener. """
        self.udp_listener.join()
