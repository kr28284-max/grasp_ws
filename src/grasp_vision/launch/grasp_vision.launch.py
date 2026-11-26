"""Launch RealSense grasp vision with the EATED eye-in-hand TF."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('grasp_vision')
    config_path = os.path.join(package_share, 'config', 'grasp_vision.yaml')
    start_realsense = LaunchConfiguration('start_realsense')
    start_rviz = LaunchConfiguration('start_rviz')
    publish_camera_mount_tf = LaunchConfiguration('publish_camera_mount_tf')

    arguments = [
        DeclareLaunchArgument('start_realsense', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument(
            'publish_camera_mount_tf',
            default_value='true',
            description='Publish end_effector_link to d435i_link static TF',
        ),
        DeclareLaunchArgument('camera_mount_x', default_value='-0.035'),
        DeclareLaunchArgument('camera_mount_y', default_value='0.0195'),
        DeclareLaunchArgument('camera_mount_z', default_value='0.048'),  # 0.052
        DeclareLaunchArgument('camera_mount_roll', default_value='0.0'),
        DeclareLaunchArgument('camera_mount_pitch', default_value='0.105'),
        DeclareLaunchArgument('camera_mount_yaw', default_value='0.0'),
    ]

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py',
            )
        ),
        condition=IfCondition(start_realsense),
        launch_arguments={
            'camera_namespace': 'realsense',
            'camera_name': 'd435i',
            'enable_color': 'true',
            'enable_depth': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'false',
            'initial_reset': 'true',
        }.items(),
    )

    camera_mount_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='d435i_mount_tf_publisher',
        output='screen',
        condition=IfCondition(publish_camera_mount_tf),
        arguments=[
            '--x', LaunchConfiguration('camera_mount_x'),
            '--y', LaunchConfiguration('camera_mount_y'),
            '--z', LaunchConfiguration('camera_mount_z'),
            '--roll', LaunchConfiguration('camera_mount_roll'),
            '--pitch', LaunchConfiguration('camera_mount_pitch'),
            '--yaw', LaunchConfiguration('camera_mount_yaw'),
            '--frame-id', 'end_effector_link',
            '--child-frame-id', 'd435i_link',
        ],
    )

    yolo_node = Node(
        package='grasp_vision',
        executable='yolo_seg_node',
        name='yolo_seg_node',
        output='screen',
        parameters=[config_path],
    )
    pose_node = Node(
        package='grasp_vision',
        executable='pose_estimation_node',
        name='pose_estimation_node',
        output='screen',
        parameters=[config_path],
    )
    table_plane_node = Node(
        package='grasp_vision',
        executable='table_plane_node',
        name='table_plane_node',
        output='screen',
        parameters=[config_path],
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='grasp_pointcloud_rviz',
        arguments=[
            '-d', os.path.join(package_share, 'rviz', 'target_pointcloud.rviz')
        ],
        output='screen',
        condition=IfCondition(start_rviz),
    )
    return LaunchDescription(
        arguments
        + [
            realsense_launch,
            camera_mount_tf,
            yolo_node,
            table_plane_node,
            pose_node,
            rviz,
        ]
    )
