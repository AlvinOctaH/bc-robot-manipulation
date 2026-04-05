from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'surgical_peg_transfer'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.world')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='Surgical peg transfer automation with ROS2 + MoveIt2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_perception_node = surgical_peg_transfer.perception.depth_perception_node:main',
            'yolo_detection_node   = surgical_peg_transfer.perception.yolo_detection_node:main',
            'task_planner_node     = surgical_peg_transfer.planning.task_planner_node:main',
            'arm_controller_node   = surgical_peg_transfer.control.arm_controller_node:main',
            'evaluation_node       = surgical_peg_transfer.evaluation.evaluation_node:main',
        ],
    },
)
