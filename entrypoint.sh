#!/bin/bash
# entrypoint.sh
#
# This runs EVERY TIME the container starts.
# It sources ROS2 and our workspace before running any command.
#
# Why do we need this?
# Environment variables set with ENV in Dockerfile persist,
# but "source" commands do NOT persist across layers —
# they only affect the current shell during build.
# This script re-sources everything in the runtime shell.

set -e

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Source our built workspace (if it exists)
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Set Gazebo model path so it can find our custom URDF
export GAZEBO_MODEL_PATH=/ros2_ws/src/surgical_peg_transfer/urdf:$GAZEBO_MODEL_PATH

# Execute whatever command was passed to docker run
# "$@" means "all arguments passed to this script"
# e.g. if you run: docker run ... ros2 launch surgical_peg_transfer full_system.launch.py
# then "$@" = "ros2 launch surgical_peg_transfer full_system.launch.py"
exec "$@"
