# ============================================================
# Dockerfile — Surgical Peg Transfer
# Base: ros:humble (Ubuntu 22.04 + ROS2 Humble)
# ============================================================

# ── Stage 1: Base image ──────────────────────────────────────
FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

# ── Stage 2: System dependencies ─────────────────────────────
RUN apt-get update && apt-get install -y \
    gazebo \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-plugins \
    ros-humble-moveit \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-ros-move-group \
    ros-humble-ur-description \
    ros-humble-ur-robot-driver \
    ros-humble-ur-moveit-config \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-pipeline \
    ros-humble-rviz2 \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    wget \
    curl \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 3: Python dependencies ─────────────────────────────
RUN pip3 install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy \
    scipy \
    matplotlib \
    pandas

# ── Stage 4: Display environment for Gazebo GUI ──────────────
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# ── Stage 5: Set up workspace ────────────────────────────────
WORKDIR /ros2_ws
COPY ros2_ws/src ./src

# ── Stage 6: Install ROS dependencies ────────────────────────
RUN rosdep update && \
    rosdep install \
        --from-paths src \
        --ignore-src \
        --rosdistro humble \
        -y

# ── Stage 7: Build the workspace ─────────────────────────────
RUN /bin/bash -c \
    "source /opt/ros/humble/setup.bash && \
     colcon build \
         --symlink-install \
         --cmake-args -DCMAKE_BUILD_TYPE=Release"

# ── Stage 8: Environment setup ───────────────────────────────
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc && \
    echo "export GAZEBO_MODEL_PATH=/ros2_ws/src/surgical_peg_transfer/urdf:$GAZEBO_MODEL_PATH" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc

# ── Entrypoint ───────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
