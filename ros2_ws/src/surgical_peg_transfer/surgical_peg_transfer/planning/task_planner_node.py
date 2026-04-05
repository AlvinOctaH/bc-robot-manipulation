import rclpy
from rclpy.node import Node

import time
from enum import Enum, auto
from dataclasses import dataclass

from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from std_msgs.msg import String, Float32


class TaskState(Enum):
    """
    All possible states of the peg transfer task.

    Why use an Enum?
    Enums prevent typos — if you write TaskState.GRASPNG Python
    throws an error immediately. Using plain strings like 'GRASPING'
    would silently fail and be hard to debug.
    """
    IDLE         = auto()
    SELECTING    = auto()
    APPROACHING  = auto()
    GRASPING     = auto()
    LIFTING      = auto()
    TRANSFERRING = auto()
    PLACING      = auto()
    RETREATING   = auto()
    DONE         = auto()


@dataclass
class PegAssignment:
    """
    Pairs a source peg with its target hole.
    dataclass auto-generates __init__, __repr__ for us.
    """
    peg_id:      int
    source:      Pose
    target:      Pose
    transferred: bool = False


class TaskPlannerNode(Node):
    """
    Finite State Machine that sequences the peg transfer task.

    How the FSM works:
      - _state_machine_step() runs at fixed rate (5Hz)
      - Each call checks current state and decides what to do
      - State transitions happen inside each handler
      - The arm controller executes the actual motion

    Why FSM and not a simple loop?
      A loop would block — the node couldn't receive new peg
      positions or handle errors while waiting for motion to complete.
      FSM lets the node stay responsive at all times.
    """

    APPROACH_HEIGHT = 0.08
    GRASP_HEIGHT    = 0.002
    LIFT_HEIGHT     = 0.10

    TARGET_HOLES = [
        [0.15,  0.05, 0.0],
        [0.15,  0.00, 0.0],
        [0.15, -0.05, 0.0],
        [0.15, -0.10, 0.0],
        [0.15, -0.15, 0.0],
        [0.15, -0.20, 0.0],
    ]

    def __init__(self):
        super().__init__('task_planner')

        self.declare_parameter('robot_frame',       'base_link')
        self.declare_parameter('max_pegs',          6)
        self.declare_parameter('state_pub_rate_hz', 5.0)

        self.robot_frame = self.get_parameter('robot_frame').value
        self.max_pegs    = self.get_parameter('max_pegs').value

        # ── State ────────────────────────────────────────────────────────
        self.state        = TaskState.IDLE
        self.assignments  = []
        self.current_idx  = 0
        self.known_pegs   = []

        # ── Subscribers ──────────────────────────────────────────────────
        self.pose_sub = self.create_subscription(
            PoseArray, '/peg_poses', self._peg_poses_callback, 10)

        # ── Publishers ───────────────────────────────────────────────────
        self.arm_goal_pub  = self.create_publisher(PoseStamped, '/arm_goal',    10)
        self.state_pub     = self.create_publisher(String,      '/task_state',  10)
        self.gripper_pub   = self.create_publisher(Float32,     '/gripper_cmd', 10)

        # ── FSM timer ────────────────────────────────────────────────────
        rate = self.get_parameter('state_pub_rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self._state_machine_step)

        self.get_logger().info('Task planner started. Waiting for peg poses...')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _peg_poses_callback(self, msg: PoseArray):
        """Update known peg positions. Trigger FSM start if idle."""
        self.known_pegs = msg.poses
        if self.state == TaskState.IDLE and len(self.known_pegs) > 0:
            self._build_assignments()
            self.state = TaskState.SELECTING
            self.get_logger().info(
                f'Detected {len(self.known_pegs)} pegs. Starting transfer.')

    # ── FSM step ──────────────────────────────────────────────────────────

    def _state_machine_step(self):
        """Called at 5Hz. Drives the FSM one step forward."""
        self._publish_state()

        handlers = {
            TaskState.IDLE:         lambda: None,
            TaskState.SELECTING:    self._select_next_peg,
            TaskState.APPROACHING:  self._go_to_approach_pose,
            TaskState.GRASPING:     self._go_to_grasp_pose,
            TaskState.LIFTING:      self._go_to_lift_pose,
            TaskState.TRANSFERRING: self._go_to_transfer_pose,
            TaskState.PLACING:      self._go_to_place_pose,
            TaskState.RETREATING:   self._retreat,
            TaskState.DONE:         self._on_done,
        }

        # Call the handler for the current state
        handlers[self.state]()

    # ── State handlers ────────────────────────────────────────────────────

    def _select_next_peg(self):
        remaining = [a for a in self.assignments if not a.transferred]
        if not remaining:
            self.state = TaskState.DONE
            return
        self.current_idx = self.assignments.index(remaining[0])
        self.state = TaskState.APPROACHING
        self.get_logger().info(
            f'Selected peg {self.assignments[self.current_idx].peg_id}')

    def _go_to_approach_pose(self):
        src = self.assignments[self.current_idx].source
        self._send_arm_goal(self._offset_pose(src, dz=self.APPROACH_HEIGHT))
        self.state = TaskState.GRASPING

    def _go_to_grasp_pose(self):
        src = self.assignments[self.current_idx].source
        self._send_arm_goal(self._offset_pose(src, dz=self.GRASP_HEIGHT))
        self._close_gripper()
        self.state = TaskState.LIFTING

    def _go_to_lift_pose(self):
        src = self.assignments[self.current_idx].source
        self._send_arm_goal(self._offset_pose(src, dz=self.LIFT_HEIGHT))
        self.state = TaskState.TRANSFERRING

    def _go_to_transfer_pose(self):
        tgt = self.assignments[self.current_idx].target
        self._send_arm_goal(self._offset_pose(tgt, dz=self.LIFT_HEIGHT))
        self.state = TaskState.PLACING

    def _go_to_place_pose(self):
        tgt = self.assignments[self.current_idx].target
        self._send_arm_goal(self._offset_pose(tgt, dz=self.GRASP_HEIGHT))
        self._open_gripper()
        self.state = TaskState.RETREATING

    def _retreat(self):
        tgt = self.assignments[self.current_idx].target
        self._send_arm_goal(self._offset_pose(tgt, dz=self.APPROACH_HEIGHT))
        self.assignments[self.current_idx].transferred = True
        self.state = TaskState.SELECTING

    def _on_done(self):
        self.get_logger().info('All pegs transferred!', once=True)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_assignments(self):
        self.assignments = []
        for i, peg_pose in enumerate(self.known_pegs[:self.max_pegs]):
            t = self.TARGET_HOLES[i % len(self.TARGET_HOLES)]
            target = Pose()
            target.position.x    = t[0]
            target.position.y    = t[1]
            target.position.z    = t[2]
            target.orientation.w = 1.0
            self.assignments.append(
                PegAssignment(peg_id=i, source=peg_pose, target=target))

    def _send_arm_goal(self, pose: Pose):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_frame
        msg.pose            = pose
        self.arm_goal_pub.publish(msg)

    def _open_gripper(self):
        self.gripper_pub.publish(Float32(data=0.0))

    def _close_gripper(self):
        self.gripper_pub.publish(Float32(data=1.0))

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state.name))

    @staticmethod
    def _offset_pose(base: Pose, dz: float = 0.0) -> Pose:
        """Return a copy of base pose shifted up by dz in Z."""
        p = Pose()
        p.position.x  = base.position.x
        p.position.y  = base.position.y
        p.position.z  = base.position.z + dz
        p.orientation = base.orientation
        return p


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()