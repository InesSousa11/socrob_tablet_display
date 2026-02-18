#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source ~/ros2_ws_tablet/install/setup.bash

echo "🚀 Launching tablet display..."
ros2 launch tablet_display tablet_display.launch.py \
  namespace:=tablet \
  fullscreen:=false \
  window_width:=800 window_height:=480 \
  window_x:=0 window_y:=0 \
  ui_rate_hz:=30.0 \
  reliability:=1