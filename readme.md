# tablet_display (ROS 2 Humble)

Lifecycle node that displays either:
- **Text** (`std_msgs/String`) on `/tablet/text`
- **Image** (`sensor_msgs/Image`) on `/tablet/image`

It renders to an **OpenCV window** sized for the robot tablet (default `800x480`) and supports multi-monitor placement (`window_x`, `window_y`).

This package also includes a small helper node `cuc_publisher` to publish text or an image file for quick testing.

---

## Package contents

- `tablet_display/tablet_display_lifecycle_node.py`  
  Lifecycle display node (OpenCV GUI)

- `tablet_display/cuc_publisher.py`  
  Publisher helper for text/images (loads image from disk, optional letterbox fit)

- `launch/tablet_display.launch.py`  
  Launches the lifecycle node inside an optional namespace

- `launch/run_tablet_display.sh`  
  Convenience script to source ROS + workspace and launch with tablet defaults

---

## Dependencies

- ROS 2 Humble
- Python packages:
  - `rclpy`, `rclpy_lifecycle`
  - `std_msgs`, `sensor_msgs`
  - `cv_bridge`
  - `opencv-python` (or system OpenCV)
  - `numpy`

---

## Build

Inside your workspace:

```bash
cd ~/ros2_ws_tablet
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source ~/ros2_ws_tablet/install/setup.bash