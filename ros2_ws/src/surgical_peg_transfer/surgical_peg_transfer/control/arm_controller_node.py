import rclpy
from rclpy.node import Node

import threading

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32

try:
    from moveit.planning import MoveItPy
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False


class ArmControllerNode(Node):
    """
    MoveIt2 interface node.

    Receives target EEF poses from the task planner
    and executes them via MoveIt2.

    Why a separate thread for execution?
      MoveIt2 planning + execution can take several seconds.
      If we ran it directly in the callback, the node would
      freeze and stop receiving new messages during that time.
      A background thread keeps the node responsive.

    Topics:
      SUB  /arm_goal     (geometry_msgs/PoseStamped)
      SUB  /gripper_cmd  (std_msgs/Float32)
      PUB  /arm_status   (std_msgs/String) MOVING / REACHED / FAILED
    """

    def __init__(self):
        super().__init__('arm_controller')

        self.declare_parameter('planning_group',   'ur_manipulator')
        self.declare_parameter('ee_link',          'tool0')
        self.declare_parameter('velocity_scaling', 0.3)
        self.declare_parameter('accel_scaling',    0.3)

        self.planning_group = self.get_parameter('planning_group').value
        self.ee_link        = self.get_parameter('ee_link').value
        self.vel_scale      = self.get_parameter('velocity_scaling').value
        self.accel_scale    = self.get_parameter('accel_scaling').value

        # ── MoveIt2 init ─────────────────────────────────────────────────
        self.moveit     = None
        self.move_group = None

        if MOVEIT_AVAILABLE:
            try:
                self.moveit     = MoveItPy(node_name='arm_controller_moveit')
                self.move_group = self.moveit.get_planning_component(
                    self.planning_group)
                self.get_logger().info(
                    f'MoveIt2 ready. Group: {self.planning_group}')
            except Exception as e:
                self.get_logger().error(f'MoveIt2 init failed: {e}')
        else:
            self.get_logger().warn(
                'MoveIt2 not available — dry-run mode active.')

        # ── State ────────────────────────────────────────────────────────
        self._lock         = threading.Lock()
        self._pending_goal = None

        # ── Subscribers ──────────────────────────────────────────────────
        self.goal_sub    = self.create_subscription(
            PoseStamped, '/arm_goal',    self._goal_callback,    10)
        self.gripper_sub = self.create_subscription(
            Float32,     '/gripper_cmd', self._gripper_callback, 10)

        # ── Publishers ───────────────────────────────────────────────────
        self.status_pub = self.create_publisher(String, '/arm_status', 10)

        # ── Background execution thread ──────────────────────────────────
        # This thread watches for pending goals and executes them.
        # It runs independently of the ROS2 spin loop.
        self._exec_thread = threading.Thread(
            target=self._execution_loop, daemon=True)
        self._exec_thread.start()

        self.get_logger().info('Arm controller node started.')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _goal_callback(self, msg: PoseStamped):
        """
        Store incoming goal for the execution thread to pick up.
        We use a lock because two threads access _pending_goal:
          - This callback (ROS2 spin thread) writes it
          - _execution_loop (background thread) reads it
        Without a lock, we could get a race condition.
        """
        with self._lock:
            self._pending_goal = msg
        self.get_logger().debug(
            f'New goal received: '
            f'({msg.pose.position.x:.3f}, '
            f'{msg.pose.position.y:.3f}, '
            f'{msg.pose.position.z:.3f})')

    def _gripper_callback(self, msg: Float32):
        """0.0 = open, 1.0 = close."""
        action = 'CLOSING' if msg.data > 0.5 else 'OPENING'
        self.get_logger().info(f'Gripper {action}')
        self._execute_gripper(msg.data)

    # ── Execution loop ────────────────────────────────────────────────────

    def _execution_loop(self):
        """
        Background thread — polls for pending goals and executes them.
        Sleeps 50ms between checks to avoid busy-waiting.
        """
        import time
        while rclpy.ok():
            goal = None
            with self._lock:
                if self._pending_goal is not None:
                    goal = self._pending_goal
                    self._pending_goal = None

            if goal is not None:
                self._execute_goal(goal)
            else:
                time.sleep(0.05)

    def _execute_goal(self, goal: PoseStamped):
        """Plan and execute motion to target EEF pose."""
        self._publish_status('MOVING')

        if self.move_group is None:
            # Dry-run mode — just log what would happen
            self.get_logger().info(
                f'[DRY-RUN] Moving to '
                f'({goal.pose.position.x:.3f}, '
                f'{goal.pose.position.y:.3f}, '
                f'{goal.pose.position.z:.3f})')
            self._publish_status('REACHED')
            return

        try:
            self.move_group.set_start_state_to_current_state()
            self.move_group.set_goal_state(
                pose_stamped_msg=goal,
                pose_link=self.ee_link,
            )

            plan = self.move_group.plan()
            if plan:
                robot  = self.moveit.get_robot()
                result = robot.execute(plan.trajectory, controllers=[])
                if result:
                    self.get_logger().info('Goal reached.')
                    self._publish_status('REACHED')
                else:
                    self.get_logger().warn('Execution failed.')
                    self._publish_status('FAILED')
            else:
                self.get_logger().warn('Planning failed.')
                self._publish_status('FAILED')

        except Exception as e:
            self.get_logger().error(f'MoveIt2 error: {e}')
            self._publish_status('FAILED')

    def _execute_gripper(self, cmd: float):
        """
        Gripper control placeholder.
        Replace with actual gripper action client when hardware is ready.
        """
        if self.move_group is None:
            action = 'CLOSE' if cmd > 0.5 else 'OPEN'
            self.get_logger().info(f'[DRY-RUN] Gripper {action}')

    def _publish_status(self, status: str):
        self.status_pub.publish(String(data=status))


def main(args=None):
    rclpy.init(args=args)
    node = ArmControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()