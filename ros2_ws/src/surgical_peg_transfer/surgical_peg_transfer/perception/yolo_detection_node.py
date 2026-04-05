import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YoloDetectionNode(Node):
    """
    YOLOv11 detection node.

    Subscribes to the camera color image, runs inference,
    and publishes detection centres as MarkerArray so the
    depth perception node can back-project them to 3D.

    Topics:
      SUB  /camera/color/image_raw  (sensor_msgs/Image)
      PUB  /detections              (visualization_msgs/MarkerArray)
      PUB  /detection_image         (sensor_msgs/Image) -- debug overlay
    """

    def __init__(self):
        super().__init__('yolo_detection')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('model_path',           'yolov11n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_topic',          '/camera/color/image_raw')

        model_path = self.get_parameter('model_path').value
        self.conf  = self.get_parameter('confidence_threshold').value
        img_topic  = self.get_parameter('image_topic').value

        # ── Load YOLO model ──────────────────────────────────────────────
        self.bridge = CvBridge()
        self.model  = None

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.get_logger().info(f'YOLOv11 loaded: {model_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load YOLO: {e}')
        else:
            self.get_logger().warn('Ultralytics not installed, stub mode.')

        # ── Subscribers ──────────────────────────────────────────────────
        self.img_sub = self.create_subscription(
            Image, img_topic, self._image_callback, 10)

        # ── Publishers ───────────────────────────────────────────────────
        self.detect_pub = self.create_publisher(
            MarkerArray, '/detections',       10)
        self.debug_pub  = self.create_publisher(
            Image,       '/detection_image',  10)

        self.get_logger().info('YOLO detection node started.')

    # ── Callback ──────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image):
        """
        Run YOLO inference on each incoming frame.

        For each detection:
          - Draw bounding box on debug image
          - Publish centre pixel as a Marker
            (depth_perception_node will use this centre
             to sample the depth image and get 3D position)
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')
            return

        if self.model is None:
            return

        results  = self.model(frame, conf=self.conf, verbose=False)
        markers  = MarkerArray()

        for result in results:
            for i, box in enumerate(result.boxes):
                # Get bounding box corners in pixel coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Calculate centre pixel of the bounding box
                # This is what we send to depth_perception_node
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Build marker — we store pixel coords in position.x/y
                # (not a real 3D pose — depth_perception reads these as pixels)
                m = Marker()
                m.header          = msg.header
                m.ns              = 'detections'
                m.id              = i
                m.type            = Marker.SPHERE
                m.action          = Marker.ADD
                m.pose.position.x = float(cx)
                m.pose.position.y = float(cy)
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0
                m.scale.x         = 10.0
                m.scale.y         = 10.0
                m.scale.z         = 10.0
                m.color           = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
                m.lifetime        = Duration(sec=1)
                markers.markers.append(m)

                # Draw on debug image
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 100, 255), -1)
                cv2.putText(
                    frame,
                    f'{result.names[int(box.cls)]} {box.conf[0]:.2f}',
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 100), 1)

        self.detect_pub.publish(markers)
        self.debug_pub.publish(
            self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()