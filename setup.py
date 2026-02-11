from setuptools import setup

package_name = "tablet_display"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/tablet_display.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ines",
    maintainer_email="ines@example.com",
    description="Lifecycle node that displays text or images on the robot tablet via an OpenCV window.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "tablet_display_lifecycle_node = tablet_display.tablet_display_lifecycle_node:main",
            "cuc_publisher = tablet_display.cuc_publisher:main",
        ],
    },
)