import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('grasp_vision')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'camera_namespace': 'camera',
            'camera_name': 'camera',
            'enable_color': 'true',
            'enable_depth': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'initial_reset': 'true',
        }.items()
    )

    yolo_node = Node(
        package='grasp_vision',
        executable='yolo_seg_node',
        name='yolo_seg_node',
        output='screen'
    )

    pose_node = Node(
        package='grasp_vision',
        executable='pose_estimation_node',
        name='pose_estimation_node',
        output='screen'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='grasp_pointcloud_rviz',
        arguments=[
            '-d',
            os.path.join(package_share, 'rviz', 'target_pointcloud.rviz')
        ],
        output='screen'
    )

    return LaunchDescription([
        realsense_launch,
        yolo_node,
        pose_node,
        rviz,
    ])
