"""
This launch file launches the 2d landmark simulator and the particle filter
"""


import os

from ament_index_python import get_package_share_directory
import launch.substitutions
from launch_ros.actions import Node
from launch.actions import Shutdown


def generate_launch_description():

    particle_filter_node = Node(
        executable='particle_filter',
        name='particle_filter',
        namespace='',
        on_exit=Shutdown(),
        package='particle_filter',
        parameters=[],
        emulate_tty=True,
        output='screen',
        remappings=[])

    landmark_sim_node = Node(
        executable='landmark_sim_2d_vis',
        name='landmark_sim_2d_vis',
        namespace='',
        on_exit=Shutdown(),
        package='landmark_sim_2d_vis',
        parameters=[],
        emulate_tty=True,
        output='screen',
        remappings=[])

    rviz_cfg_path = os.path.join(get_package_share_directory('particle_filter'),
                                 'rviz/particle_filter.rviz')

    rviz_node = Node(
        executable='rviz2',
        name='rviz2',
        namespace='',
        on_exit=Shutdown(),
        package='rviz2',
        parameters=[],
        arguments=['-d', str(rviz_cfg_path)],
        remappings=[
        ])

    return launch.LaunchDescription([
        landmark_sim_node,
        particle_filter_node,
        rviz_node
    ])
