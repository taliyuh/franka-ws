FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV RCUTILS_COLORIZED_OUTPUT=1

SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# System + ROS 2 + Franka + Simulation + Control Dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # General development tools
    git \
    curl \
    wget \
    vim \
    nano \
    bash-completion \
    build-essential \
    cmake \
    pkg-config \
    # ROS workspace tools & linters
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    python3-colcon-common-extensions \
    ros-humble-ament-cmake-clang-format \
    ros-humble-ament-clang-format \
    clang-format \
    # ROS 2 Control suite (includes diff_drive_controller, joint_state_broadcaster, etc.)
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-diff-drive-controller \
    # Simulation & Gazebo integration
    ros-humble-gz-ros2-control \
    ros-humble-ros-gz \
    # Franka & MoveIt base packages
    ros-humble-moveit \
    ros-humble-geometric-shapes \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    # libfranka runtime/build deps (libfranka 0.20 links Pinocchio 4 + Poco 1.11)
    ros-humble-pinocchio \
    libpoco-dev \
    libfmt-dev \
    libeigen3-dev \
    # Remaining workspace build/exec deps. This list is what
    #   rosdep install --from-paths src --ignore-src -r --simulate
    # reports inside the container; re-run it after adding packages to src/.
    ros-humble-ament-cmake-clang-tidy \
    ros-humble-hardware-interface-testing \
    ros-humble-camera-info-manager \
    ros-humble-diagnostic-updater \
    ros-humble-image-publisher \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    python3-requests \
    # franka_mobile / franka_mobile_sensors (TMR base: joystick, RealSense, SICK)
    ros-humble-joy \
    ros-humble-teleop-twist-joy \
    ros-humble-teleop-twist-keyboard \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-sick-safetyscanners2 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Azure Kinect Sensor SDK
# ---------------------------------------------------------------------------
# Microsoft publishes the Linux SDK packages in its Ubuntu 18.04 repository.
# The 1.4.2 packages work with this Ubuntu 22.04 ROS image and include the
# proprietary depth engine required at runtime. k4a-tools is intentionally not
# installed because its legacy libsoundio1 dependency is unavailable on Jammy;
# the ROS driver only needs the runtime and development packages below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && install -d -m 0755 /usr/share/keyrings \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
       | gpg --dearmor --yes -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/18.04/prod bionic main" \
       > /etc/apt/sources.list.d/microsoft-prod.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
       libk4a1.4 \
       libk4a1.4-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Workspace Setup
# ---------------------------------------------------------------------------
WORKDIR /franka_ws

# Initialize rosdep (run if /etc/ros/rosdep does not already exist)
RUN if [ ! -d /etc/ros/rosdep ]; then rosdep init; fi && rosdep update

# Ensure ROS environment is automatically sourced in interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "if [ -f /franka_ws/install/setup.bash ]; then source /franka_ws/install/setup.bash; fi" >> ~/.bashrc

CMD ["/bin/bash"]
