import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration


class DepthPerceptionNode(Node):
    """
    Converts 2D peg detections + depth image into 3D poses.

    Pipeline:
      1. Receive 2D bounding boxes from YOLO node (/detections)
      2. For each detection, sample the depth image at the box centre
      3. Back-project pixel (u, v, d) -> camera-frame (X, Y, Z)
      4. Publish as PoseArray on /peg_poses
    """

    def __init__(self):
        super().__init__('depth_perception')

        # ── Parameters ──────────────────────────────────────────────────
        # These can be overridden from camera_params.yaml at launch time
        self.declare_parameter('depth_topic',  '/camera/depth/image_raw')
        self.declare_parameter('info_topic',   '/camera/color/camera_info')
        self.declare_parameter('detect_topic', '/detections')
        self.declare_parameter('depth_scale',  0.001)
        self.declare_parameter('min_depth',    0.05)
        self.declare_parameter('max_depth',    1.50)
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')

        depth_topic  = self.get_parameter('depth_topic').value
        info_topic   = self.get_parameter('info_topic').value
        detect_topic = self.get_parameter('detect_topic').value

        self.depth_scale = self.get_parameter('depth_scale').value
        self.min_depth   = self.get_parameter('min_depth').value
        self.max_depth   = self.get_parameter('max_depth').value

        # ── State ────────────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self.K            = None   # 3x3 camera intrinsic matrix
        self.latest_depth = None   # most recent depth frame

        # ── QoS ──────────────────────────────────────────────────────────
        # Sensors publish with BEST_EFFORT reliability (they drop frames
        # rather than queue them). We must match this or we receive nothing.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ──────────────────────────────────────────────────
        self.depth_sub  = self.create_subscription(
            Image,       depth_topic,  self._depth_callback,     sensor_qos)
        self.info_sub   = self.create_subscription(
            CameraInfo,  info_topic,   self._info_callback,      sensor_qos)
        self.detect_sub = self.create_subscription(
            MarkerArray, detect_topic, self._detection_callback, 10)

        # ── Publishers ───────────────────────────────────────────────────
        self.pose_pub = self.create_publisher(PoseArray,    '/peg_poses',     10)
        self.viz_pub  = self.create_publisher(MarkerArray,  '/peg_poses_viz', 10)

        self.get_logger().info('Depth perception node started.')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _info_callback(self, msg: CameraInfo):
        """
        Store camera intrinsics. Only needs to run once.
        K is the 3x3 intrinsic matrix:
          [fx  0  cx]
          [ 0 fy  cy]
          [ 0  0   1]
        fx, fy = focal lengths in pixels
        cx, cy = principal point (optical centre) in pixels
        """
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f'Camera intrinsics received. '
                f'fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f}')

    def _depth_callback(self, msg: Image):
        """Cache the latest depth image for use in detection callback."""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth conversion failed: {e}')

    def _detection_callback(self, msg: MarkerArray):
        """
        For each 2D detection, back-project its centre pixel to 3D.
        Publishes a PoseArray — one Pose per detected peg.
        """
        if self.latest_depth is None:
            self.get_logger().warn('No depth image yet, skipping.')
            return
        if self.K is None:
            self.get_logger().warn('No camera intrinsics yet, skipping.')
            return

        poses_3d = []

        for marker in msg.markers:
            # YOLO node publishes detection centre as marker position
            u = int(marker.pose.position.x)  # pixel column
            v = int(marker.pose.position.y)  # pixel row

            point_3d = self._pixel_to_3d(u, v)
            if point_3d is not None:
                pose = Pose()
                pose.position.x   = point_3d[0]
                pose.position.y   = point_3d[1]
                pose.position.z   = point_3d[2]
                pose.orientation.w = 1.0
                poses_3d.append(pose)

        if poses_3d:
            pose_array = PoseArray()
            pose_array.header.stamp    = self.get_clock().now().to_msg()
            pose_array.header.frame_id = self.get_parameter('camera_frame').value
            pose_array.poses           = poses_3d
            self.pose_pub.publish(pose_array)
            self._publish_viz(pose_array)
            self.get_logger().debug(f'Published {len(poses_3d)} peg poses.')

    # ── Core geometry ──────────────────────────────────────────────────────

    def _pixel_to_3d(self, u: int, v: int):
        """
        Back-project pixel (u, v) to 3D point using pinhole camera model.

        Pinhole model equations:
            Z = depth value in metres
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

        We sample a 5x5 patch around the pixel and take the median depth
        to reduce noise from a single noisy pixel reading.
        """
        h, w = self.latest_depth.shape[:2]

        # Check pixel is inside image bounds
        if not (0 <= u < w and 0 <= v < h):
            return None

        # Sample 5x5 patch, take median of valid (non-zero) readings
        patch = self.latest_depth[
            max(0, v-2):min(h, v+3),
            max(0, u-2):min(w, u+3)
        ]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        Z = float(np.median(valid)) * self.depth_scale

        # Reject depth readings outside valid range
        if not (self.min_depth <= Z <= self.max_depth):
            return None

        # Back-project using intrinsics
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return np.array([X, Y, Z])

    # ── Visualisation ──────────────────────────────────────────────────────

    def _publish_viz(self, pose_array: PoseArray):
        """Publish green sphere markers so we can see detections in RViz."""
        markers = MarkerArray()
        for i, pose in enumerate(pose_array.poses):
            m = Marker()
            m.header        = pose_array.header
            m.ns            = 'peg_3d'
            m.id            = i
            m.type          = Marker.SPHERE
            m.action        = Marker.ADD
            m.pose          = pose
            m.scale.x       = 0.015
            m.scale.y       = 0.015
            m.scale.z       = 0.015
            m.color         = ColorRGBA(r=0.2, g=0.8, b=0.4, a=0.9)
            m.lifetime      = Duration(sec=1)
            markers.markers.append(m)
        self.viz_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = DepthPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()