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
        gazebo,
        ur5_state_publisher,
        spawn_ur5,
        spawn_peg_board,
        rviz,
    ])