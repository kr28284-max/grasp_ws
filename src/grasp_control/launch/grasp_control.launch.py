"""Launch the grasp controller with EATED hardware parameters."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('grasp_control'),
        'config',
        'grasp_control.yaml',
    )
    config_argument = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Grasp controller parameter YAML file',
    )
    controller_node = Node(
        package='grasp_control',
        executable='control_node',
        name='grasp_control_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )
    router_node = Node(
        package='grasp_control',
        executable='grasp_router_node',
        name='grasp_router_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )
    dummy_node = Node(
        package='grasp_control',
        executable='rb3_dummy_executor_node',
        name='rb3_dummy_executor_node',
        output='screen',
    )
    return LaunchDescription([
        config_argument,
        controller_node,
        router_node,
        dummy_node,
    ])
