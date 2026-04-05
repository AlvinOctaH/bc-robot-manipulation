import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg = get_package_share_directory('surgical_peg_transfer')
    ur_description_pkg = get_package_share_directory('ur_description')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    ur_type      = LaunchConfiguration('ur_type',      default='ur5')
    yolo_model   = LaunchConfiguration('yolo_model',   default='yolov11n.pt')
    log_results  = LaunchConfiguration('log_results',  default='true')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': os.path.join(pkg, 'worlds', 'peg_transfer_world.world'),
            'verbose': 'false',
            'pause':   'false',
        }.items(),
    )

    ur5_urdf = Command([
    'xacro ',
    os.path.join(ur_description_pkg, 'urdf', 'ur.urdf.xacro'),
    ' name:=ur',
    ' ur_type:=', ur_type,
    ' use_fake_hardware:=true',
    ' fake_sensor_commands:=true',
    ' sim_gazebo:=true',
    ])

    ur5_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ur5_state_publisher',
        parameters=[{
            'robot_description': ur5_urdf,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    spawn_ur5 = TimerAction(
        period=2.0,
        actions=[Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_ur5',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'ur5',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.0',
            ],
            output='screen',
        )]
    )

    spawn_peg_board = TimerAction(
        period=3.0,
        actions=[Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_peg_board',
            arguments=[
                '-file',   os.path.join(pkg, 'urdf', 'peg_board.urdf'),
                '-entity', 'peg_board',
                '-x', '0.50',
                '-y', '0.00',
                '-z', '0.7775',
            ],
            output='screen',
        )]
    )

    depth_perception_node = Node(
        package='surgical_peg_transfer',
        executable='depth_perception_node',
        name='depth_perception',
        parameters=[
            os.path.join(pkg, 'config', 'camera_params.yaml'),
            {'use_sim_time': use_sim_time},
        ],
        output='screen',
    )

    yolo_detection_node = Node(
        package='surgical_peg_transfer',
        executable='yolo_detection_node',
        name='yolo_detection',
        parameters=[{
            'model_path': yolo_model,
            'use_sim_time': use_sim_time,
            'confidence_threshold': 0.5,
        }],
        output='screen',
    )

    task_planner_node = TimerAction(
        period=5.0,
        actions=[Node(
            package='surgical_peg_transfer',
            executable='task_planner_node',
            name='task_planner',
            parameters=[
                os.path.join(pkg, 'config', 'planner_params.yaml'),
                {'use_sim_time': use_sim_time},
            ],
            output='screen',
        )]
    )

    arm_controller_node = TimerAction(
        period=5.0,
        actions=[Node(
            package='surgical_peg_transfer',
            executable='arm_controller_node',
            name='arm_controller',
            parameters=[
                os.path.join(pkg, 'config', 'robot_params.yaml'),
                {'use_sim_time': use_sim_time},
            ],
            output='screen',
        )]
    )

    evaluation_node = Node(
        package='surgical_peg_transfer',
        executable='evaluation_node',
        name='evaluation',
        parameters=[{
            'use_sim_time': use_sim_time,
            'log_results': log_results,
            'log_dir': '/tmp/peg_transfer_logs',
        }],
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('ur_type',      default_value='ur5'),
        DeclareLaunchArgument('yolo_model',   default_value='yolov11n.pt'),
        DeclareLaunchArgument('log_results',  default_value='true'),
        gazebo,
        ur5_state_publisher,
        spawn_ur5,
        spawn_peg_board,
        depth_perception_node,
        yolo_detection_node,
        task_planner_node,
        arm_controller_node,
        evaluation_node,
        rviz,
    ])