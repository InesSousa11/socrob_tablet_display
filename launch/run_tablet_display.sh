#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source ~/ros2_ws_tablet/install/setup.bash

# Tablet monitor origin (from: xrandr --query, XWAYLAND1 800x480+1920+0)
WIN_X=1920
WIN_Y=0

echo "🚀 Launching tablet display (fullscreen on tablet @ ${WIN_X},${WIN_Y})..."
ros2 launch tablet_display tablet_display.launch.py \
  namespace:=tablet \
  fullscreen:=true \
  window_width:=800 window_height:=480 \
  window_x:=${WIN_X} window_y:=${WIN_Y} \
  ui_rate_hz:=30.0 \
  reliability:=1