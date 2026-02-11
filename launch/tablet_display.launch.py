#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("text_topic", default_value="/tablet/text"),
        DeclareLaunchArgument("image_topic", default_value="/tablet/image"),
        DeclareLaunchArgument("fullscreen", default_value="True"),
        DeclareLaunchArgument("window_width", default_value="800"),
        DeclareLaunchArgument("window_height", default_value="480"),
        DeclareLaunchArgument("window_x", default_value="0"),
        DeclareLaunchArgument("window_y", default_value="0"),
        DeclareLaunchArgument("ui_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("reliability", default_value="1"),  # 1 reliable, 2 best effort

        LifecycleNode(
            package="tablet_display",
            executable="tablet_display_lifecycle_node",
            name="tablet_display",
            output="screen",
            parameters=[{
                "text_topic": LaunchConfiguration("text_topic"),
                "image_topic": LaunchConfiguration("image_topic"),
                "fullscreen": LaunchConfiguration("fullscreen"),
                "window_width": LaunchConfiguration("window_width"),
                "window_height": LaunchConfiguration("window_height"),
                "window_x": LaunchConfiguration("window_x"),
                "window_y": LaunchConfiguration("window_y"),
                "ui_rate_hz": LaunchConfiguration("ui_rate_hz"),
                "reliability": LaunchConfiguration("reliability"),
            }],
        ),
    ])